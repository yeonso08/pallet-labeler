"""Native Tk smoke check of focused XZ rendering; closes only its own window."""
from pathlib import Path
import tkinter as tk
import numpy as np
from matplotlib.backend_bases import MouseEvent,MouseButton
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
        # Direction no longer isolates on its own; its button does.
        app.toggle_isolation();root.update()
        app.side_view();root.update();app.cancel_settle();app.settle()
        assert not app.view3d and app.side and app.isolated
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
            # Turning the cloud keeps what was chosen.
            assert app.selected is not None and app.isolated
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
        # A drag selects in 3D too, and the projection has to land on the canvas:
        # points are picked through the projection, not the 3D axes' own limits.
        app.cancel_settle();app.settle();root.update()
        pixels=app.ax.transData.transform(app.screen_xy())
        width,height=app.canvas.get_width_height()
        assert pixels.min(0).min()>-width and pixels.max(0).max()<2*max(width,height)
        lo=pixels[app.labels==3].min(0);hi=pixels[app.labels==3].max(0)
        corners=[(lo[0]-2,lo[1]-2),(hi[0]+2,lo[1]-2),(hi[0]+2,hi[1]+2),(lo[0]-2,hi[1]+2)]
        MouseEvent('button_press_event',app.canvas,*corners[0],button=MouseButton.LEFT)._process()
        for x,y in corners[1:]:
            MouseEvent('motion_notify_event',app.canvas,x,y,button=MouseButton.LEFT)._process()
        assert app.lasso._selection_artist.get_visible()
        MouseEvent('button_release_event',app.canvas,*corners[0],button=MouseButton.LEFT)._process()
        root.update()
        assert app.selected is not None and app.selected.sum()>=(app.labels==3).sum()
        app.clear_selection();app.set_direction('3D');root.update()
        app.set_direction('2D');root.update()
        assert not app.view3d and app.ax.name!='3d'
        assert app.view_buttons['2D'].cget('style')=='On.TButton'
        assert app.view_buttons['위'].cget('style')=='On.TButton'
        # A plane remembers the zoom it was left at.
        app.set_direction('앞');root.update()
        app.ax.set_xlim(-11.,13.);app.ax.set_ylim(-17.,19.)
        app.set_direction('위');root.update()
        assert app.ax.get_xlim()!=(-11.,13.)
        app.set_direction('앞');root.update()
        assert app.ax.get_xlim()==(-11.,13.) and app.ax.get_ylim()==(-17.,19.)
        app.fit_view();root.update()
        assert app.ax.get_xlim()!=(-11.,13.)
        # Keys reach the view, except while a number is being typed.
        root.focus_force();root.update()
        for key,plane in (('2','앞'),('3','옆'),('1','위')):
            root.event_generate(f'<Key-{key}>');root.update()
            assert app.direction==plane,(key,app.direction)
        root.event_generate('<Key-Tab>');root.update();assert app.view3d
        root.event_generate('<Key-Tab>');root.update();assert not app.view3d
        app.label_entry.focus_set();root.update()
        root.event_generate('<Key-2>');root.update()
        assert app.direction=='위'
        app.cancel_settle()
        print('Tk render, focused XZ data, height lasso, kept selection, 3D lasso, remembered zoom, shortcuts and v5 model selection OK')
    finally:
        root.destroy()


if __name__=='__main__':main()
