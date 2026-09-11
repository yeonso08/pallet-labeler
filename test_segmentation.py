import unittest
from unittest.mock import patch
import numpy as np
from engine import segment_prediction, infer


class PalletPreservationTests(unittest.TestCase):
    def scene(self):
        p=np.full((30,30),.95)
        p[10:20,10:20]=.01
        r=dict(shape=p.shape,valid=np.ones(p.shape,bool),flat=np.arange(p.size))
        return r,p,np.zeros(p.shape)

    def test_observed_pallet_hole_remains_pallet(self):
        r,p,b=self.scene()
        legacy=segment_prediction(r,p,b)[3]
        protected=segment_prediction(r,p,b,hole_fill_probability=.2)[3]
        self.assertTrue(np.all(legacy>1))
        self.assertTrue(np.all(protected[10:20,10:20]==1))
        self.assertTrue(np.all(protected[:10]>1))

    def test_small_unobserved_hole_still_bridged(self):
        r,p,b=self.scene()
        p[:]=.95;p[15,15]=0;r['valid'][15,15]=False
        labels=segment_prediction(r,p,b,hole_fill_probability=.2)[3]
        self.assertGreater(labels[15,15],1)

    def test_all_pallet_stays_pallet(self):
        r,p,b=self.scene();p[:]=.01
        self.assertTrue(np.all(segment_prediction(r,p,b,hole_fill_probability=.2)[0]==1))

    def test_infer_uses_model_setting(self):
        r,p,b=self.scene()
        with patch('engine.predict_grid',return_value=(r,p,b)):
            self.assertEqual(infer(None,{'hole_fill_probability':.2})[3][15,15],1)
            self.assertGreater(infer(None,{})[3][15,15],1)

    def test_infer_dispatches_optional_region_linker(self):
        r,p,b=self.scene();classifier=object()
        with patch('engine.predict_grid',return_value=(r,p,b)), patch('region_affinity.link_prediction',return_value='linked') as link:
            self.assertEqual(infer(None,{'region_linker':classifier,'region_link_threshold':.6}),'linked')
            self.assertIs(link.call_args.args[1],classifier)
            self.assertEqual(link.call_args.args[2],.6)

    def test_larger_seed_filter_preserves_isolated_small_object(self):
        r,p,b=self.scene();p[:]=.01;p[2:15,2:15]=.95
        labels=segment_prediction(r,p,b,hole_fill_probability=.5,min_seed_cells=600)[3]
        self.assertTrue(np.all(labels[2:15,2:15]>1))
        self.assertEqual(labels[20,20],1)

    def test_small_internal_seed_does_not_split_one_object(self):
        r,p,b=self.scene();p[:]=.95;b[:,8:10]=.5
        original=segment_prediction(r,p,b,min_seed_cells=150)[4]
        filtered=segment_prediction(r,p,b,min_seed_cells=300)[4]
        self.assertEqual(original['objects'],2)
        self.assertEqual(filtered['objects'],1)

    def test_smoothing_removes_thin_noise_but_keeps_wide_boundary(self):
        r,p,b=self.scene();p[:]=.95;b[:,15]=.5
        self.assertEqual(segment_prediction(r,p,b)[4]['objects'],2)
        self.assertEqual(segment_prediction(r,p,b,seed_smoothing=3)[4]['objects'],1)
        b[:,14:17]=.5
        self.assertEqual(segment_prediction(r,p,b,seed_smoothing=3)[4]['objects'],2)


if __name__=='__main__':unittest.main()
