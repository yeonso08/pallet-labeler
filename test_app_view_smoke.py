"""Native Tk smoke check of focused XZ rendering; closes only its own window."""
from pathlib import Path
import tkinter as tk
import numpy as np
from app import App
from engine import read_cloud,LABEL


def main():
    root=tk.Tk()
    try:
        app=App(root);root.update()
        path=Path(__file__).resolve().parent/'results/region_link_v5/example/401_labeled.ply'
        app.ply,app.xyz=read_cloud(path);app.current=path
        app.labels=app.ply['vertex'].data[LABEL].astype(np.int32)
        app.review=np.zeros(len(app.labels));app.labels_version+=1
        app.focus=3;app.selected=app.labels==3;app.pick_highlight=True
        app.side_view();root.update();app.cancel_settle();app.settle()
        assert not app.view3d and app.side
        expected=app.xyz[app.labels==3][:,[0,2]]
        displayed=np.asarray(app.full_artist.get_offsets())
        assert len(displayed)==len(expected)
        np.testing.assert_allclose(displayed,expected)
        assert app.ax.get_aspect()=='auto'
        assert app.view_buttons['앞'].cget('style')=='On.TButton'
        assert app.active_model_name=='가림·조각 연결 개선 (v5)'
        # Partial height selection must use XZ and only the focused label.
        lo=expected.min(0);hi=expected.max(0);middle=(lo[1]+hi[1])/2
        app.select([(lo[0]-1,lo[1]-1),(hi[0]+1,lo[1]-1),(hi[0]+1,middle),(lo[0]-1,middle)])
        root.update()
        assert app.selected is not None and np.all(app.labels[app.selected]==3)
        assert np.all(app.xyz[app.selected,2]<=middle)
        out=Path(__file__).resolve().parent/'results/ui_merge_check';out.mkdir(parents=True,exist_ok=True)
        app.fig.savefig(out/'side_view.png',dpi=110)
        for name,columns in [('옆',[1,2]),('위',[0,1])]:
            app.set_direction(name);root.update()
            np.testing.assert_allclose(np.asarray(app.full_artist.get_offsets()),app.xyz[app.labels==3][:,columns])
        app.set_direction('앞');root.update()
        app.clip_enabled.set(True);app.change_clip_axis()
        app.clip_low.set(25.);app.clip_high.set(75.);app.apply_clip();root.update()
        visible=app.visible_mask()
        assert visible.any() and np.sum(visible)<np.sum(app.labels==3)
        np.testing.assert_allclose(np.asarray(app.full_artist.get_offsets()),app.xyz[visible][:,[0,2]])
        # A broad lasso cannot select hidden points, even with Shift-add.
        app.press_key='shift';app.selected=app.labels==3
        app.select([(lo[0]-1,lo[1]-1),(hi[0]+1,lo[1]-1),(hi[0]+1,hi[1]+1),(lo[0]-1,hi[1]+1)])
        assert not np.any(app.selected&~visible)
        app.fig.savefig(out/'clipped_front.png',dpi=110)
        app.toggle_isolation();root.update()
        assert app.direction=='앞' and not app.isolated
        app.reset_clip();root.update()
        assert not app.clip_enabled.get()
        app.clear_selection();root.update()
        assert app.focus is None and not app.side
        app.toggle();root.update()
        assert app.view3d and app.ax.name=='3d'
        assert app.view_buttons['3D'].cget('style')=='On.TButton'
        assert app.view_buttons['2D'].cget('style')=='Off.TButton'
        app.set_direction('2D');root.update()
        assert not app.view3d and app.ax.name!='3d'
        assert app.view_buttons['2D'].cget('style')=='On.TButton'
        assert app.view_buttons['앞'].cget('style')=='On.TButton'
        app.cancel_settle()
        print('Tk render, focused XZ data, height lasso, clear, 2D/3D return and v5 model selection OK')
    finally:
        root.destroy()


if __name__=='__main__':main()
