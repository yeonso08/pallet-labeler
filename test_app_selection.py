"""Editing regressions: stacked labels, side projection and focus lifecycle."""
from pathlib import Path
from unittest.mock import Mock,patch
import unittest
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from app import App


class Value:
    def __init__(self,value):self.value=value
    def get(self):return self.value
    def set(self,value):self.value=value


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.a=App.__new__(App);a=self.a
        a.xyz=np.array([[0,0,0],[0,0,5],[0,0,10],[1,1,5],[1,1,10]],float)
        a.labels=np.array([1,2,3,2,3]);a.review=np.zeros(5)
        a.selected=None;a.focus=None;a.side=False;a.view3d=False;a.busy=False
        a.pick_highlight=False;a.press_key=None;a.show_pallet=Value(True);a.status=Value('')
        a.label=Value('4');a.labels_version=0;a.undo=[];a.reset_view=False
        a.direction='위';a.isolated=False;a.views={}
        a.fig=Figure(figsize=(6,4));canvas=FigureCanvasAgg(a.fig);a.ax=a.fig.add_subplot()
        a.ax.set_xlim(-2,2);a.ax.set_ylim(-2,12);canvas.draw()
        a.draw=Mock()

    def test_xy_lasso_excludes_stacked_pallet_and_other_object(self):
        a=self.a;a.focus=2
        a.select([(-1,-1),(2,-1),(2,2),(-1,2)])
        np.testing.assert_array_equal(a.selected,[False,True,False,True,False])
        a.assign()
        np.testing.assert_array_equal(a.labels,[1,4,3,4,3])

    def test_shift_lasso_cannot_reintroduce_other_labels(self):
        a=self.a;a.focus=2;a.selected=np.ones(5,bool);a.press_key='shift'
        a.select([(-1,-1),(2,-1),(2,2),(-1,2)])
        self.assertTrue(np.all(a.labels[a.selected]==2))

    def test_side_uses_xz_and_same_focus(self):
        a=self.a;a.focus=2;a.side_view()
        self.assertFalse(a.view3d);self.assertTrue(a.side)
        np.testing.assert_array_equal(a.screen_xy(),a.xyz[:,[0,2]])
        a.select([(-.5,4),(.5,4),(.5,6),(-.5,6)])
        np.testing.assert_array_equal(a.selected,[False,True,False,False,False])
        a.assign();self.assertEqual(a.focus,2)
        a.rollback();np.testing.assert_array_equal(a.labels,[1,2,3,2,3])

    def test_side_pick_cannot_choose_hidden_other_object(self):
        a=self.a;a.focus=2;a.side=True;a.isolated=True
        a.pick((0,10))
        self.assertIsNone(a.selected);self.assertEqual(a.focus,2);self.assertTrue(a.side)

    def test_pick_hidden_pallet_filters_before_nearest_search(self):
        a=self.a;a.show_pallet.set(False)
        a.pick((0,0))
        self.assertEqual(a.focus,2);self.assertTrue(np.all(a.labels[a.selected]==2))

    def test_clear_leaves_side_and_releases_focus(self):
        a=self.a;a.focus=2;a.side=True
        a.clear_selection()
        self.assertIsNone(a.focus);self.assertFalse(a.side);self.assertTrue(a.reset_view)

    def test_save_remaps_focus_with_compacted_ids(self):
        a=self.a;a.labels=np.array([1,4,8,4,8]);a.focus=8;a.side=True
        a.current=Path('/original/input.ply');a.files=[a.current];a.out=Value('/tmp');a.ply=object()
        with patch('app.filedialog.asksaveasfilename',return_value='/tmp/selection-test.ply'),patch('app.save_cloud') as save,patch('app.messagebox.showinfo'):
            a.save()
        self.assertEqual(a.focus,3)
        np.testing.assert_array_equal(a.labels,[1,2,3,2,3]);save.assert_called_once()

    def clipping(self):
        a=self.a;a.focus=2;a.isolated=True;a.clip_enabled=Value(True)
        a.clip_bounds=(0,0.,1.);a.clip_low=Value(0.);a.clip_high=Value(50.)
        return a

    def test_clip_hides_and_prevents_selection_of_same_label_points(self):
        a=self.clipping()
        a.select([(-1,-1),(2,-1),(2,2),(-1,2)])
        np.testing.assert_array_equal(a.selected,[False,True,False,False,False])
        a.assign();np.testing.assert_array_equal(a.labels,[1,4,3,2,3])

    def test_hidden_old_selection_cannot_be_applied(self):
        a=self.clipping();a.selected=a.labels==2
        a.assign();np.testing.assert_array_equal(a.labels,[1,4,3,2,3])

    def test_direction_switches_plane_and_leaves_isolation_alone(self):
        a=self.a;a.focus=2
        for name,plane in [('위',(0,1)),('앞',(0,2)),('옆',(1,2))]:
            a.set_direction(name);self.assertFalse(a.isolated)
            self.assertEqual(a.plane(),plane)
        a.set_direction('3D');self.assertTrue(a.view3d);self.assertFalse(a.isolated)
        a.isolated=True
        for name in ('위','앞','옆','3D','2D'):
            a.set_direction(name);self.assertTrue(a.isolated)

    def test_view_changes_keep_the_selection(self):
        a=self.a;a.focus=2;a.selected=a.labels==2;a.pick_highlight=True
        for name in ('앞','옆','3D','2D','위'):
            a.set_direction(name)
            self.assertIsNotNone(a.selected);self.assertTrue(a.pick_highlight)
        chosen=self.clipping();chosen.selected=chosen.labels==2
        chosen.clip_low.set(10.);chosen.apply_clip()
        np.testing.assert_array_equal(chosen.selected,chosen.labels==2)

    def test_hidden_points_are_reported_not_applied(self):
        a=self.clipping();a.selected=a.labels>0
        visible=a.visible_mask()
        a.assign()
        self.assertIn('숨겨진',a.status.get())
        np.testing.assert_array_equal(a.labels[visible],[4])
        np.testing.assert_array_equal(a.labels[~visible],[1,3,2,3])

    def test_2d_returns_to_the_plane_used_before_3d(self):
        a=self.a;a.focus=None;a.isolated=False
        a.set_direction('옆');a.set_direction('3D')
        self.assertTrue(a.view3d);self.assertFalse(a.side);self.assertEqual(a.plane(),(0,1))
        a.set_direction('2D')
        self.assertFalse(a.view3d);self.assertTrue(a.side);self.assertEqual(a.plane(),(1,2))
        a.toggle();self.assertTrue(a.view3d)
        a.toggle();self.assertEqual(a.plane(),(1,2))

    def test_reversed_clip_bounds_and_empty_range_are_safe(self):
        a=self.clipping();a.clip_low.set(100.);a.clip_high.set(75.)
        np.testing.assert_array_equal(a.visible_mask(),[False,False,False,True,False])
        a.clip_low.set(25.);a.clip_high.set(75.)
        self.assertFalse(a.visible_mask().any())


if __name__=='__main__':unittest.main()
