import argparse
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.roi_manifest import MARKERS, build_manifest, digest
from src.evaluate_ultimate import evaluate_locked
from src.models.anchored_ihc import AnchoredIHC
from src.models.marker_context import reconstruction_loss
from src.select_ultimate import apply_deployment, choose_markers, select
from src.train_marker_context import infer_command, load_model, seed_all, train


class UltimateTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)

    def test_zero_init_exactly_preserves_baseline_and_freezes_it(self):
        net=AnchoredIHC({'width':4,'markers':4,'context':True},width=4)
        net.train()
        self.assertFalse(net.baseline.training)
        self.assertTrue(all(not p.requires_grad for p in net.baseline.parameters()))
        x=torch.rand(2,1,31,35)
        with torch.no_grad():
            before={k:v.clone() for k,v in net.baseline.state_dict().items()}
            base=net.baseline(x)
        torch.testing.assert_close(net(x),base,rtol=0,atol=0)
        optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-3)
        loss=reconstruction_loss(net(x),torch.rand_like(base))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in net.baseline.parameters()))
        self.assertTrue(all(d.head.weight.grad.abs().sum()>0 for d in net.refiner.decoders))
        optimizer.step()
        for name,value in net.baseline.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)
        self.assertGreater((net(x)-base).abs().sum().item(),0)

    def test_residual_bound_and_marker_fallback(self):
        net=AnchoredIHC({'width':4,'markers':4,'context':True},width=4,residual_limit=.15).eval()
        for decoder in net.refiner.decoders:
            torch.nn.init.constant_(decoder.head.bias,10.)
        x=torch.rand(1,1,32,32)
        with torch.no_grad():
            base,candidate=net.baseline(x),net(x)
            self.assertLessEqual((candidate-base).abs().max().item(),.150001)
            net.deployment_mask=[True,False,True,False]
            deployed=net(x)
            torch.testing.assert_close(deployed[:,[1,3]],base[:,[1,3]],rtol=0,atol=0)
            torch.testing.assert_close(deployed[:,[0,2]],candidate[:,[0,2]],rtol=0,atol=0)

    def test_selection_requires_both_metrics_and_rejects_nan(self):
        baseline={'count':3,'markers':{m:{'ssim':.7,'psnr':23.} for m in MARKERS}}
        candidate=copy.deepcopy(baseline)
        candidate['markers'][MARKERS[0]].update(ssim=.71,psnr=23.1)
        candidate['markers'][MARKERS[1]].update(ssim=.71,psnr=22.9)
        candidate['markers'][MARKERS[2]].update(ssim=.69,psnr=24.)
        self.assertEqual(choose_markers(baseline,candidate),[True,False,False,False])
        candidate['markers'][MARKERS[0]]['ssim']=float('nan')
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            choose_markers(baseline,candidate)

    def make_fixture(self,root):
        for marker in ('DAPI',)+MARKERS:
            folder=root/'train'/marker
            folder.mkdir(parents=True)
            for roi in range(4):
                arr=np.random.default_rng(roi).integers(0,80,(256,256),dtype=np.uint8)
                Image.fromarray(arr).save(folder/f'ROI{roi:03}_00_00.jpg')
        build_manifest(root,root/'split.json',val_rois=1,holdout_rois=1)
        return argparse.Namespace(data_root=str(root),manifest=str(root/'split.json'),
            output=str(root/'base'),seed=42,device='cpu',batch_size=1,width=4,epochs=1,lr=5e-4,
            no_context=False,no_cache=False,architecture='context',val_jpeg_quality=95,
            train_limit=1,val_limit=1,resume=None)

    def test_train_resume_select_and_locked_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            args=self.make_fixture(root)
            train(args)
            base,base_ck=load_model(root/'base/best.pt',torch.device('cpu'))
            args.output=str(root/'refiner')
            args.architecture='anchored'
            args.baseline_checkpoint=str(root/'base/best.pt')
            train(args)
            net,ck=load_model(root/'refiner/best.pt',torch.device('cpu'))
            for name,value in base.state_dict().items():
                torch.testing.assert_close(net.baseline.state_dict()[name],value,rtol=0,atol=0)
            args.baseline_checkpoint=None
            args.resume=str(root/'refiner/last.pt')
            train(args)
            selection_args=argparse.Namespace(data_root=str(root),manifest=str(root/'split.json'),
                checkpoint=str(root/'refiner/best.pt'),output=str(root/'deployment.json'),
                device='cpu',batch_size=1,seed=42,tta=4,jpeg_quality=95,limit=1,no_cache=False,allow_smoke=True)
            report=select(selection_args)
            self.assertTrue(report['smoke_only'])
            for marker in MARKERS:
                for metric in ('ssim','psnr'):
                    self.assertGreaterEqual(report['deployed']['markers'][marker][metric]+2e-6,
                                            report['baseline']['markers'][marker][metric])
            with self.assertRaisesRegex(ValueError,'Smoke'):
                apply_deployment(net,ck,selection_args.checkpoint,root/'deployment.json',4,95)
            with self.assertRaisesRegex(ValueError,'TTA'):
                apply_deployment(net,ck,selection_args.checkpoint,root/'deployment.json',8,95,True)
            holdout_args=argparse.Namespace(data_root=str(root),manifest=str(root/'split.json'),
                checkpoint=selection_args.checkpoint,deployment=str(root/'deployment.json'),
                output=str(root/'holdout.json'),device='cpu',batch_size=1,seed=42,no_cache=False,allow_smoke=True)
            held=evaluate_locked(holdout_args)
            self.assertEqual(held['split'],'holdout')
            self.assertEqual(held['deployed']['count'],1)
            self.assertEqual(held['deployment_sha256'],report['sha256'])
            with self.assertRaises(FileExistsError):
                evaluate_locked(holdout_args)
            holdout_args.output=str(root/'leaky_holdout.json')
            leaked=copy.deepcopy(ck)
            leaked['run']['train_names']+=json.loads((root/'split.json').read_text())['splits']['holdout']
            torch.save(leaked,root/'leaked_candidate.pt')
            holdout_args.checkpoint=str(root/'leaked_candidate.pt')
            from src.select_ultimate import sha256
            leaked_report=copy.deepcopy(report)
            leaked_report['checkpoint_sha256']=sha256(holdout_args.checkpoint)
            leaked_report['sha256']=digest({k:v for k,v in leaked_report.items() if k!='sha256'})
            (root/'leaked_deployment.json').write_text(json.dumps(leaked_report))
            holdout_args.deployment=str(root/'leaked_deployment.json')
            with self.assertRaisesRegex(ValueError,'used for training'):
                evaluate_locked(holdout_args)
            inputs=root/'test/DAPI'
            inputs.mkdir(parents=True)
            Image.fromarray(np.zeros((256,256),dtype=np.uint8)).save(inputs/'ROI025_00_00.jpg')
            infer=argparse.Namespace(checkpoint=selection_args.checkpoint,input=str(inputs),output=str(root/'pred'),
                deployment=str(root/'deployment.json'),device='cpu',seed=42,batch_size=1,tta=4,jpeg_quality=95,
                allow_smoke=True)
            infer_command(infer)
            self.assertEqual(len(list((root/'pred/results/test').glob('*/*_fake.jpg'))),4)
            broken=copy.deepcopy(report)
            broken['checkpoint_sha256']='wrong'
            broken['sha256']=digest({k:v for k,v in broken.items() if k!='sha256'})
            (root/'bad.json').write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError,'checkpoint mismatch'):
                apply_deployment(net,ck,selection_args.checkpoint,root/'bad.json',4,95,True)
            broken=copy.deepcopy(report)
            broken['inference_source_sha256']={}
            broken['sha256']=digest({k:v for k,v in broken.items() if k!='sha256'})
            (root/'changed_source.json').write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError,'source changed'):
                apply_deployment(net,ck,selection_args.checkpoint,root/'changed_source.json',4,95,True)
            # Full-data checkpoints cannot be reused as held-out baselines.
            base_ck['run']['train_names']+=json.loads((root/'split.json').read_text())['splits']['val']
            torch.save(base_ck,root/'leaky.pt')
            args.resume=None
            args.baseline_checkpoint=str(root/'leaky.pt')
            args.output=str(root/'leaky_run')
            with self.assertRaisesRegex(ValueError,'already saw'):
                train(args)

    def test_complete_public_launcher_on_fixture(self):
        import zipfile
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.make_fixture(root)
            inputs=root/'test/DAPI'
            inputs.mkdir(parents=True)
            Image.fromarray(np.zeros((256,256),dtype=np.uint8)).save(inputs/'ROI025_00_00.jpg')
            repo=Path(__file__).resolve().parents[1]
            common=[sys.executable,str(repo/'scripts/run_final.py'),'--data-root',str(root),
                    '--manifest',str(root/'split.json'),'--run-dir',str(root/'full'),
                    '--device','cpu','--batch-size','1','--baseline-width','4','--width','4',
                    '--baseline-epochs','1','--refine-epochs','1','--tta','1']
            def invoke(*options,check=True):
                result=subprocess.run(common+list(options),cwd=repo,capture_output=True,text=True)
                if check:
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return result
            invoke('--mode','train')
            invoke('--mode','train','--resume')
            missing=invoke('--mode','predict','--output',str(root/'submission'),check=False)
            self.assertNotEqual(missing.returncode,0)
            self.assertIn('evaluate once',missing.stderr)
            invoke('--mode','evaluate')
            invoke('--mode','predict','--output',str(root/'submission'))
            with zipfile.ZipFile(root/'submission/submission.zip') as archive:
                self.assertEqual(len(archive.namelist()),4)
                self.assertTrue(all(name.startswith('results/test/') for name in archive.namelist()))
            provenance=json.loads((root/'submission/provenance.json').read_text())
            decision=json.loads((root/'full/deployment.json').read_text())
            self.assertEqual(provenance['deployment_sha256'],decision['sha256'])


if __name__=='__main__':
    unittest.main()
