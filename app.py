"""Desktop interface: local prediction, point selection, correction and safe export."""
from pathlib import Path
import json,logging,queue,threading,time,traceback
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import numpy as np
import joblib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as Polygon
from mpl_toolkits.mplot3d import proj3d
from engine import read_cloud,infer,save_cloud,LABEL
from viewer import BACKGROUND,display_indices,point_colors
import theme

ROOT=Path(__file__).resolve().parent
# Widening one axis to hold the scale equal is exactly what the flat views ask
# for, and matplotlib says so on every draw of one.
logging.getLogger('matplotlib.axes._base').addFilter(lambda record:'fixed data aspect' not in record.getMessage())
# Matplotlib rasterizes every point on every frame, so a moving view is only as
# smooth as the point count: ~2us/point here. Swap to this many while the camera
# moves, then settle back to full detail once it stops. Marker size costs almost
# nothing, so the sparse pass draws fatter points to stay readable.
INTERACTIVE_POINTS=20000
INTERACTIVE_POINT_SCALE=2.2
SETTLE_MS=220
# A lasso this small in pixels was a click, not a drag.
CLICK_SLOP=6
PICK_RADIUS=18

class Toolbar(NavigationToolbar2Tk):
 # Panning and zooming are on the mouse now; keep only what the mouse cannot do.
 toolitems=[item for item in NavigationToolbar2Tk.toolitems if item[0] in ('Home','Save')]

class App:
 def __init__(self,win):
  self.win=win;win.title('Pallet Labeler 2.2');win.geometry('1250x850')
  self.jobs=queue.Queue();self.busy=False;self.ply=None;self.labels=None;self.selected=None;self.current=None;self.model=None
  self.files=[];self.undo=[];self.view3d=True;self.reset_view=True
  self.focus=None;self.side=False;self.pick_highlight=False
  self.isolated=False;self.direction='위';self.clip_bounds=None;self.clip_timer=None
  self.full_artist=None;self.lod_artist=None;self.moving=False;self.settle_timer=None;self.labels_version=0;self.centers=None
  self.press_key=None;self.pan_from=None;self.views={};self.canvas_size=None;self.resize_timer=None
  config_path=ROOT/'model_config.json'
  self.model_options=json.loads(config_path.read_text()) if config_path.exists() else {'기본 모델':{'file':'model.joblib','split':.1}}
  self.active_model_name=next(iter(self.model_options));self.active_model_file=ROOT/self.model_options[self.active_model_name]['file']
  fonts=theme.apply(win)
  head=ttk.Frame(win,padding=(18,12));head.pack(fill='x')
  ttk.Label(head,text='Pallet Labeler',style='Wordmark.TLabel').pack(side='left')
  ttk.Label(head,text='팔레트 1  ·  부자재 2, 3, 4…  ·  자동 결과는 확인 후 저장',style='Muted.TLabel').pack(side='left',padx=20)
  body=ttk.Panedwindow(win,orient='horizontal');body.pack(fill='both',expand=True,padx=16)
  left=ttk.Frame(body,width=265,padding=8);right=ttk.Frame(body);body.add(left,weight=0);body.add(right,weight=1)
  ttk.Label(left,text='자동 라벨링 모델',style='Muted.TLabel').pack(anchor='w')
  self.model_choice=tk.StringVar(value=self.active_model_name)
  self.model_combo=ttk.Combobox(left,textvariable=self.model_choice,values=list(self.model_options),state='readonly',width=25);self.model_combo.pack(fill='x',pady=(0,6))
  self.model_combo.bind('<<ComboboxSelected>>',self.change_model)
  ttk.Button(left,text='PLY 파일 추가',command=self.add_files).pack(fill='x')
  ttk.Button(left,text='폴더에서 추가',command=self.add_folder).pack(fill='x',pady=4)
  self.listbox=tk.Listbox(left,width=27,height=10,selectmode='extended',exportselection=False);self.listbox.pack(fill='x',pady=8)
  theme.style_listbox(self.listbox,fonts)
  self.listbox.bind('<Double-1>',lambda _:self.open_selected())
  # Delete on a full keyboard, Backspace on a Mac keyboard.
  self.listbox.bind('<Delete>',self.drop_files)
  self.listbox.bind('<BackSpace>',self.drop_files)
  ttk.Button(left,text='선택 파일 열기 / 자동 라벨링',command=self.open_selected).pack(fill='x')
  ttk.Button(left,text='선택 파일 목록에서 빼기',command=self.drop_files).pack(fill='x',pady=4)
  ttk.Label(left,text='물체 경계 분리 기준',style='Muted.TLabel').pack(anchor='w',pady=(12,0))
  self.split=tk.DoubleVar(value=self.model_options[self.active_model_name]['split'])
  theme.slider(left,self.split,.02,.35,.01).pack(fill='x')
  ttk.Label(left,text='왼쪽: 더 잘게 분리 / 오른쪽: 더 크게 합침',style='Muted.TLabel').pack(anchor='w')
  self.out=tk.StringVar(value=str(ROOT/'results'))
  ttk.Button(left,text='선택 파일 다시 자동 라벨링',command=lambda:self.open_selected(force=True)).pack(fill='x',pady=4)
  ttk.Label(left,text='저장 폴더',style='Muted.TLabel').pack(anchor='w',pady=(12,0))
  ttk.Entry(left,textvariable=self.out,width=27).pack(fill='x')
  ttk.Button(left,text='저장 폴더 선택',command=self.choose_out).pack(fill='x',pady=4)
  ttk.Button(left,text='추가한 파일 일괄 처리',command=self.batch).pack(fill='x',pady=6)
  ttk.Label(left,text='일괄 결과는 자동 초안입니다.\nCloudCompare에서 확인하세요.\n원본 파일은 변경하지 않습니다.',style='Muted.TLabel',wraplength=245).pack(anchor='w',pady=8)
  controls=ttk.Frame(right);controls.pack(fill='x')
  self.label=tk.StringVar(value='2')
  ttk.Label(controls,text='선택한 점 → 라벨',style='Muted.TLabel').pack(side='left')
  self.label_entry=ttk.Entry(controls,textvariable=self.label,width=5);self.label_entry.pack(side='left')
  ttk.Button(controls,text='적용',command=self.assign).pack(side='left',padx=3)
  ttk.Button(controls,text='새 물체로 분리',command=self.new_object).pack(side='left')
  ttk.Button(controls,text='되돌리기',command=self.rollback).pack(side='left',padx=3)
  ttk.Button(controls,text='수정 결과 저장',style='Accent.TButton',command=self.save).pack(side='right')
  display=ttk.Frame(right);display.pack(fill='x',pady=5)
  self.color_mode=tk.StringVar(value='원본 색상');self.density=tk.StringVar(value='촘촘하게');self.point_size=tk.DoubleVar(value=2.0);self.show_pallet=tk.BooleanVar(value=True);self.show_numbers=tk.BooleanVar(value=True)
  self.color_combo=ttk.Combobox(display,textvariable=self.color_mode,values=['물체별 색상','원본 색상','높이 색상'],state='readonly',width=12);self.color_combo.pack(side='left');self.color_combo.bind('<<ComboboxSelected>>',lambda _:self.draw())
  self.density_combo=ttk.Combobox(display,textvariable=self.density,values=['빠르게','촘촘하게','전체 점'],state='readonly',width=10);self.density_combo.pack(side='left',padx=5);self.density_combo.bind('<<ComboboxSelected>>',lambda _:self.draw())
  self.pallet_btn=self.switch(display,'팔레트 표시',self.show_pallet);self.pallet_btn.pack(side='left')
  self.number_btn=self.switch(display,'번호 표시',self.show_numbers);self.number_btn.pack(side='left',padx=4)
  ttk.Label(display,text='점 크기',style='Muted.TLabel').pack(side='left')
  size=theme.slider(display,self.point_size,.5,5,.1,length=75);size.pack(side='left');size.bind('<ButtonRelease-1>',lambda _:self.draw())
  ttk.Button(display,text='화면 맞춤',command=self.fit_view).pack(side='right')
  ttk.Button(display,text='선택 해제',command=self.clear_selection).pack(side='right',padx=4)
  edit=ttk.Frame(right);edit.pack(fill='x',pady=(2,4))
  self.isolate_button=ttk.Button(edit,text='선택 부재만 보기',command=self.toggle_isolation);self.isolate_button.pack(side='left')
  self.view_buttons={}
  for name in ('2D','위','앞','옆','3D'):
   button=ttk.Button(edit,text=name,command=lambda n=name:self.set_direction(n),width=4);button.pack(side='left',padx=2);self.view_buttons[name]=button
  self.stretch=tk.BooleanVar(value=True)
  ttk.Checkbutton(edit,text='두께 확대',variable=self.stretch,command=self.draw).pack(side='left',padx=4)
  self.scope_status=tk.StringVar(value='부재를 클릭하세요')
  ttk.Label(edit,textvariable=self.scope_status,style='Muted.TLabel').pack(side='left',padx=4)
  clip=ttk.Frame(right);clip.pack(fill='x',pady=(0,4))
  self.clip_enabled=tk.BooleanVar(value=False);self.clip_axis=tk.StringVar(value='높이 Z')
  self.clip_low=tk.DoubleVar(value=0.);self.clip_high=tk.DoubleVar(value=100.)
  ttk.Checkbutton(clip,text='구간 좁히기',variable=self.clip_enabled,command=self.change_clip_axis).pack(side='left')
  combo=ttk.Combobox(clip,textvariable=self.clip_axis,values=('높이 Z','깊이 Y','가로 X'),state='readonly',width=7);combo.pack(side='left');combo.bind('<<ComboboxSelected>>',self.change_clip_axis)
  self.clip_scales=[]
  for name,var in (('시작',self.clip_low),('끝',self.clip_high)):
   ttk.Label(clip,text=name,style='Muted.TLabel').pack(side='left',padx=(5,0))
   slider=ttk.Scale(clip,from_=0,to=100,variable=var,length=100,command=self.queue_clip);slider.pack(side='left');self.clip_scales.append(slider)
  self.clip_status=tk.StringVar(value='전체 구간')
  ttk.Label(clip,textvariable=self.clip_status,style='Muted.TLabel').pack(side='left',padx=4)
  ttk.Button(clip,text='구간 초기화',command=self.reset_clip).pack(side='right')
  ttk.Label(right,text='부재 클릭 → 방향 선택 → 구간 좁히기 → 드래그 선택 · 보기를 바꿔도 선택은 그대로입니다\n1 2 3: 위·앞·옆 · Tab: 2D↔3D · Shift: 추가 · Alt: 빼기 · 오른쪽 드래그: 회전/이동 · Esc: 선택 해제',style='Muted.TLabel').pack(anchor='w')
  self.fig=Figure(figsize=(8,6),facecolor=BACKGROUND);self.ax=self.fig.add_subplot(111)
  self.canvas=FigureCanvasTkAgg(self.fig,master=right);self.canvas.get_tk_widget().pack(fill='both',expand=True)
  self.toolbar=Toolbar(self.canvas,right);self.lasso=None
  theme.style_toolbar(self.toolbar,fonts)
  # add='+', or this replaces the backend's own handler and the figure stops
  # following the widget.
  self.canvas.get_tk_widget().bind('<Configure>',self.on_resize,add='+')
  self.canvas.mpl_connect('scroll_event',self.zoom)
  self.canvas.mpl_connect('button_press_event',self.start_move)
  self.canvas.mpl_connect('motion_notify_event',self.pan_move)
  self.canvas.mpl_connect('button_release_event',self.end_move)
  win.bind('<Escape>',lambda _:self.clear_selection())
  for key,name in (('1','위'),('2','앞'),('3','옆')):
   win.bind(key,self.shortcut(lambda n=name:self.set_direction(n)))
  win.bind('<Tab>',self.shortcut(self.toggle))
  self.status=tk.StringVar(value='PLY 파일을 추가하고 자동 라벨링을 시작하세요. 제공된 스캐너 좌표와 단위를 사용합니다.')
  ttk.Label(win,textvariable=self.status,style='Muted.TLabel',padding=12,wraplength=1200).pack(fill='x')
  win.after(100,self.poll);self.draw()
 def switch(self,parent,text,flag):
  """On/off control: clam's own checkbutton marks the checked state with an X."""
  button=ttk.Button(parent,text=text,style='On.TButton' if flag.get() else 'Off.TButton')
  def flip():
   flag.set(not flag.get());button.configure(style='On.TButton' if flag.get() else 'Off.TButton');self.draw()
  button.configure(command=flip)
  return button
 def shortcut(self,fn):
  """Window-wide key: let a text field keep the keys it needs to type."""
  def handler(event=None):
   widget=self.win.focus_get()
   if widget is not None and widget.winfo_class() in ('Entry','TEntry'):return
   fn();return 'break'
  return handler
 def change_model(self,event=None):
  if self.busy:self.model_choice.set(self.active_model_name);return
  self.active_model_name=self.model_choice.get();option=self.model_options[self.active_model_name]
  self.active_model_file=ROOT/option['file'];self.model=None;self.split.set(option['split'])
  self.status.set('모델을 변경했습니다. 선택 파일 다시 자동 라벨링을 누르면 적용됩니다.')
 def add_files(self):
  self.add(filedialog.askopenfilenames(filetypes=[('Point cloud','*.ply')]))
 def add_folder(self):
  p=filedialog.askdirectory()
  if p:self.add(sorted(Path(p).glob('*.ply')))
 def add(self,paths):
  for p in paths:
   p=Path(p)
   if p not in self.files:self.files.append(p);self.listbox.insert('end',p.name)
  if self.files and not self.listbox.curselection():self.listbox.selection_set(0)
 def drop_files(self,event=None):
  if self.busy:self.status.set('처리 중에는 목록을 바꿀 수 없습니다.');return
  chosen=list(self.listbox.curselection())
  if not chosen:return
  dropping=[self.files[i] for i in chosen]
  closing=self.current in dropping
  if closing and self.undo and not messagebox.askyesno('미저장 수정','열려 있는 파일을 목록에서 빼면 화면에서도 닫히고 수정 내용이 사라집니다. 계속할까요?'):return
  for i in reversed(chosen):self.listbox.delete(i);self.files.pop(i)
  if self.files:
   nxt=min(chosen[0],len(self.files)-1);self.listbox.selection_set(nxt);self.listbox.see(nxt)
  if closing:self.close_current()
  self.status.set(f'목록에서 {len(chosen)}개를 뺐습니다. 디스크의 파일은 그대로입니다.')
 def close_current(self):
  """Forget the open cloud so the canvas matches the list again."""
  self.current=self.ply=self.xyz=self.labels=self.review=self.info=None
  self.focus=None;self.side=False;self.pick_highlight=False
  self.isolated=False;self.reset_clip(False);self.views.clear()
  self.selected=None;self.undo=[];self.centers=None;self.labels_version+=1;self.reset_view=True
  self.draw()
 def choose_out(self):
  p=filedialog.askdirectory()
  if p:self.out.set(p)
 def run(self,fn):
  if self.busy:messagebox.showinfo('처리 중','현재 작업이 끝날 때까지 기다려 주세요.');return
  self.busy=True;self.status.set('자동 라벨링 중… 파일 크기에 따라 잠시 걸립니다.')
  def worker():
   try:fn()
   except Exception as e:self.jobs.put(('error',str(e)));traceback.print_exc()
   finally:self.jobs.put(('done',None))
  threading.Thread(target=worker,daemon=True).start()
 def get_model(self):
  if self.model is None:self.model=joblib.load(self.active_model_file)
  return self.model
 def open_selected(self,force=False):
  s=self.listbox.curselection()
  if not s:return
  if self.undo and not messagebox.askyesno('미저장 수정','현재 수정 내용을 저장하지 않고 다른 파일을 열까요?'):return
  path=self.files[s[0]];split=float(self.split.get())
  def work():
   ply,xyz=read_cloud(path)
   a=ply['vertex'].data
   if LABEL in a.dtype.names and not force:
    v=a[LABEL]
    if not np.isfinite(v).all() or np.any(v<1) or np.any(v!=np.floor(v)):raise ValueError('라벨 값이 유효하지 않습니다. 다시 자동 라벨링을 사용하세요.')
    labels=v.astype(np.int32);review=a['scalar_review_needed'].copy() if 'scalar_review_needed' in a.dtype.names else np.zeros(len(a),np.float32)
    info=dict(objects=len(np.unique(labels[labels>1])),review_fraction=float(review.mean()))
   else:labels,review,r,g,info=infer(ply,self.get_model(),split=split)
   self.jobs.put(('loaded',(path,ply,xyz,labels,review,info)))
  self.run(work)
 def unique_path(self,directory,stem):
  directory=Path(directory).expanduser();p=directory/f'{stem}_labeled.ply';i=2
  while p.exists():p=directory/f'{stem}_labeled_{i}.ply';i+=1
  return p
 def batch(self):
  if not self.files:return
  files=list(self.files);out=self.out.get();split=float(self.split.get())
  def work():
   rows=[]
   for i,path in enumerate(files):
    self.jobs.put(('status',f'일괄 처리 {i+1}/{len(files)} · {path.name}'))
    try:
     ply,_=read_cloud(path);labels,review,_,_,info=infer(ply,self.get_model(),split=split)
     dest=self.unique_path(out,path.stem);save_cloud(ply,labels,dest,review)
     rows.append(dict(file=str(path),output=str(dest),status='auto_draft',**info))
    except Exception as e:rows.append(dict(file=str(path),error=str(e)))
   directory=Path(out).expanduser();directory.mkdir(parents=True,exist_ok=True)
   report=directory/f'batch_{time.time_ns()}.json';report.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
   failed=sum('error' in r for r in rows);self.jobs.put(('status',f'일괄 처리 완료: {len(rows)-failed}개 저장, {failed}개 실패 · {out}'))
  self.run(work)
 def poll(self):
  try:
   while True:
    kind,data=self.jobs.get_nowait()
    if kind=='done':self.busy=False
    elif kind=='error':self.status.set('오류: '+data);messagebox.showerror('오류',data)
    elif kind=='status':self.status.set(data)
    elif kind=='loaded':
     self.current,self.ply,self.xyz,self.labels,self.review,self.info=data;self.selected=None;self.undo=[];self.reset_view=True;self.labels_version+=1
     self.focus=None;self.side=False;self.pick_highlight=False
     self.isolated=False;self.reset_clip(False);self.views.clear()
     self.status.set(f'{self.current.name} · {len(self.labels):,}점 · 부자재 {self.info["objects"]}개 추정 · 경계/불확실 점 {self.info["review_fraction"]:.1%} · 자동 초안');self.draw()
  except queue.Empty:pass
  self.win.after(100,self.poll)
 def on_resize(self,event):
  """The rectangle that catches the mouse is shaped from the canvas, so redraw once it settles."""
  size=(event.width,event.height)
  if size==self.canvas_size:return
  self.canvas_size=size
  if self.resize_timer is not None:self.win.after_cancel(self.resize_timer)
  self.resize_timer=self.win.after(200,self.settle_resize)
 def settle_resize(self):
  self.resize_timer=None;self.draw()
 def fit_view(self):self.views.pop(self.view_key(),None);self.reset_view=True;self.draw()
 def plane(self):return ((1,2) if getattr(self,'direction','앞')=='옆' else (0,2)) if self.side else (0,1)
 def reset_clip(self,redraw=True):
  if getattr(self,'clip_timer',None) is not None:self.win.after_cancel(self.clip_timer);self.clip_timer=None
  self.clip_bounds=None
  if hasattr(self,'clip_enabled'):
   self.clip_enabled.set(False);self.clip_low.set(0.);self.clip_high.set(100.);self.clip_status.set('전체 구간')
  if redraw:self.draw()
 def change_clip_axis(self,event=None):
  if self.busy:return
  if self.focused_label() is None:
   self.reset_clip(False);self.status.set('먼저 부재를 클릭해서 고르세요.');return
  axis={'높이 Z':2,'깊이 Y':1,'가로 X':0}[self.clip_axis.get()]
  values=self.xyz[self.labels==self.focus,axis]
  self.clip_bounds=(axis,float(values.min()),float(values.max()))
  self.clip_low.set(0.);self.clip_high.set(100.);self.apply_clip()
 def queue_clip(self,value=None):
  if self.busy or not self.clip_enabled.get():return
  if self.clip_timer is not None:self.win.after_cancel(self.clip_timer)
  self.clip_timer=self.win.after(80,self.apply_clip)
 def apply_clip(self):
  # The selection survives; assign() drops whatever the slice hides.
  self.clip_timer=None;self.draw()
 def visible_mask(self):
  mask=np.ones(len(self.labels),bool)
  focus=self.focused_label()
  if getattr(self,'isolated',False) and focus is not None:mask&=self.labels==focus
  if not self.show_pallet.get():mask&=self.labels!=1
  if getattr(self,'clip_bounds',None) is not None and self.clip_enabled.get():
   axis,lo,hi=self.clip_bounds
   a,b=sorted((self.clip_low.get(),self.clip_high.get()))
   low=lo+(hi-lo)*a/100;high=lo+(hi-lo)*b/100
   mask&=(self.xyz[:,axis]>=low)&(self.xyz[:,axis]<=high)
  return mask
 def toggle_isolation(self):
  if self.busy or self.labels is None:return
  if self.focused_label() is None:self.status.set('먼저 부재를 클릭해서 고르세요.');return
  self.isolated=not self.isolated
  # What the view is fitted to changes, so the remembered zooms no longer apply.
  self.views.clear();self.reset_view=True;self.draw()
 def view_key(self):return '3D' if self.view3d else self.direction
 def remember_view(self):
  """Keep this plane's zoom so coming back to it looks the way it was left."""
  # Nothing to remember from a canvas that is not showing the cloud.
  if self.full_artist is None or (getattr(self.ax,'name','')=='3d')!=self.view3d:return
  if self.view3d:self.views['3D']=((self.ax.elev,self.ax.azim,self.ax.roll),(self.ax.get_xlim(),self.ax.get_ylim(),self.ax.get_zlim()))
  else:self.views[self.direction]=(None,(self.ax.get_xlim(),self.ax.get_ylim()))
 def set_direction(self,name):
  if self.busy:return
  # '2D' is the counterpart of '3D': it leaves the rotating view for the flat
  # plane that was last in use, so the chosen plane survives a trip through 3D.
  if name=='2D':name=self.direction
  self.remember_view()
  if name=='3D':self.view3d=True;self.side=False
  else:self.direction=name;self.view3d=False;self.side=name in ('앞','옆')
  # Turning the cloud changes nothing about what is selected or hidden, so the
  # selection stays and isolation is left to its own button.
  self.reset_view=True;self.draw()
 def focused_label(self):
  if self.focus is not None and (self.labels is None or not np.any(self.labels==self.focus)):self.focus=None
  return self.focus
 def side_view(self):
  self.set_direction('위' if self.side else '앞')
 def clear_selection(self):
  if self.selected is None and self.focus is None and not self.side:return
  self.selected=None;self.focus=None;self.pick_highlight=False
  self.isolated=False;self.reset_clip(False);self.views.clear()
  if self.side:self.side=False;self.direction='위';self.reset_view=True
  self.status.set('선택과 부재 고정을 해제했습니다.');self.draw()
 def zoom(self,event):
  if event.inaxes!=self.ax or self.labels is None:return
  factor=.85 if event.button=='up' else 1/.85
  axes=['x','y','z'] if self.view3d else ['x','y']
  for axis in axes:
   lo,hi=getattr(self.ax,'get_'+axis+'lim')();center=(lo+hi)/2;half=(hi-lo)*factor/2
   getattr(self.ax,'set_'+axis+'lim')(center-half,center+half)
  self.show_moving();self.settle_later()
 def start_move(self,event):
  if event.inaxes!=self.ax or self.labels is None:return
  self.press_key=event.key
  # Left drag is the lasso and leaves the camera alone; the other buttons move the view.
  if event.button not in (2,3):return
  if not self.view3d:
   origin=self.ax.transData.inverted().transform([(event.x,event.y),(event.x+1,event.y+1)])
   self.pan_from=(event.x,event.y,self.ax.get_xlim(),self.ax.get_ylim(),origin[1]-origin[0])
  self.show_moving()
 def pan_move(self,event):
  if self.pan_from is None or event.x is None:return
  x0,y0,xlim,ylim,scale=self.pan_from
  dx=(x0-event.x)*scale[0];dy=(y0-event.y)*scale[1]
  self.ax.set_xlim(xlim[0]+dx,xlim[1]+dx);self.ax.set_ylim(ylim[0]+dy,ylim[1]+dy)
  self.canvas.draw_idle()
 def end_move(self,event):
  self.pan_from=None
  if self.moving:self.settle_later()
 def show_moving(self):
  self.cancel_settle()
  if self.lod_artist is not None and not self.moving:
   self.moving=True;self.full_artist.set_visible(False);self.lod_artist.set_visible(True)
  self.canvas.draw_idle()
 def settle_later(self):
  self.cancel_settle();self.settle_timer=self.win.after(SETTLE_MS,self.settle)
 def cancel_settle(self):
  if self.settle_timer is not None:self.win.after_cancel(self.settle_timer);self.settle_timer=None
 def settle(self):
  self.settle_timer=None
  if not self.moving:return
  self.moving=False
  if self.lod_artist is not None:self.lod_artist.set_visible(False);self.full_artist.set_visible(True)
  self.canvas.draw_idle()
 def add_lod(self,pts,colors,size):
  if len(pts)<=INTERACTIVE_POINTS:return
  k=display_indices(len(pts),INTERACTIVE_POINTS);p=pts[k];c=colors[k];size=size*INTERACTIVE_POINT_SCALE
  if self.view3d:self.lod_artist=self.ax.scatter(p[:,0],p[:,1],p[:,2],c=c,s=size,depthshade=False,linewidths=0,antialiased=False)
  else:
   h,v=self.plane();self.lod_artist=self.ax.scatter(p[:,h],p[:,v],c=c,s=size,linewidths=0,antialiased=False)
  self.lod_artist.set_visible(False)
 def label_centers(self):
  # Scanning every point once per object is too slow to repeat on each redraw.
  if self.centers is None or self.centers[0]!=self.labels_version:
   centers={}
   for lab in np.unique(self.labels):
    v=self.xyz[self.labels==lab];c=np.median(v,axis=0);c[2]=np.quantile(v[:,2],.9)+3;centers[int(lab)]=c
   self.centers=(self.labels_version,centers)
  return self.centers[1]
 def draw(self):
  focus=self.focused_label()
  if focus is None and (self.isolated or self.clip_bounds is not None):
   self.isolated=False;self.reset_clip(False);self.reset_view=True
  self.isolate_button.configure(text='전체 함께 보기' if self.isolated else '선택 부재만 보기')
  active={'3D'} if self.view3d else {'2D',self.direction}
  for name,button in self.view_buttons.items():button.configure(style='On.TButton' if name in active else 'Off.TButton')
  self.scope_status.set(f'고정: 라벨 {focus}' if focus is not None else '부재를 클릭하세요')
  for slider in self.clip_scales:slider.configure(state='normal' if self.clip_enabled.get() else 'disabled')
  if self.clip_bounds is not None and self.clip_enabled.get():
   axis,lo,hi=self.clip_bounds;a,b=sorted((self.clip_low.get(),self.clip_high.get()))
   self.clip_status.set(f'{lo+(hi-lo)*a/100:.1f} ~ {lo+(hi-lo)*b/100:.1f}')
  else:self.clip_status.set('전체 구간')
  camera=None;limits=None
  if self.labels is not None:
   if not self.reset_view:
    if self.view3d and getattr(self.ax,'name','')=='3d':camera=(self.ax.elev,self.ax.azim,self.ax.roll);limits=(self.ax.get_xlim(),self.ax.get_ylim(),self.ax.get_zlim())
    elif not self.view3d and getattr(self.ax,'name','')!='3d':limits=(self.ax.get_xlim(),self.ax.get_ylim())
   elif self.views.get(self.view_key()) is not None:camera,limits=self.views[self.view_key()]
  if self.lasso:self.lasso.disconnect_events();self.lasso=None
  self.cancel_settle();self.moving=False;self.full_artist=None;self.lod_artist=None
  # The mouse only reaches points inside the axes rectangle, so that rectangle
  # has to cover the whole canvas. mplot3d insists on a square one, so hand it
  # a square that contains the canvas rather than one that fits inside it, and
  # take the extra size back out of the zoom.
  width,height=self.canvas.get_width_height();square=max(width,height)
  fills=bool(width and height)
  rect=[(width-square)/2/width,(height-square)/2/height,square/width,square/height] if self.view3d and fills else [0,0,1,1]
  self.fig.clear();self.fig.set_facecolor(BACKGROUND);self.ax=self.fig.add_axes(rect,projection='3d' if self.view3d else None)
  self.ax.set_facecolor(BACKGROUND);self.ax.set_axis_off()
  if self.labels is None:
   self.fig.text(.5,.5,'왼쪽에서 PLY 파일을 추가한 뒤 열어 주세요',ha='center',va='center',color=theme.MUTED,fontsize=11)
  else:
   budget={'빠르게':60000,'촘촘하게':200000,'전체 점':None}[self.density.get()]
   base=np.flatnonzero(self.visible_mask());ix=base[display_indices(len(base),budget)]
   fit=self.xyz[self.labels==focus] if self.isolated and focus is not None else self.xyz
   pts=self.xyz[ix];colors=point_colors(self.ply,self.xyz,self.labels,ix,self.color_mode.get(),self.selected,self.pick_highlight)
   size=float(self.point_size.get())
   if self.view3d:
    self.ax.set_proj_type('ortho')
    # The left button draws the lasso, so rotating moves to the right button.
    self.ax.mouse_init(rotate_btn=3,pan_btn=2,zoom_btn=[])
    self.full_artist=self.ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=colors,s=size,depthshade=False,linewidths=0,antialiased=False)
    self.add_lod(pts,colors,size)
    extent=np.maximum(np.ptp(fit,axis=0),1)
    self.ax.set_box_aspect(extent,zoom=1.12*(.98*min(width,height)/square if fills else 1))
    self.ax.view_init(*(camera or (55,-65,0)))
    if limits:
     self.ax.set_xlim(limits[0]);self.ax.set_ylim(limits[1]);self.ax.set_zlim(limits[2])
    else:
     for k,axis in enumerate('xyz'):
      lo,hi=np.quantile(fit[:,k],[0,1]);pad=max((hi-lo)*.025,.5);getattr(self.ax,'set_'+axis+'lim')(lo-pad,hi+pad)
   else:
    h,v=self.plane()
    self.full_artist=self.ax.scatter(pts[:,h],pts[:,v],c=colors,s=size,linewidths=0,antialiased=False);self.add_lod(pts,colors,size)
    # Equal scaling shrinks the axes box by default, which would put part of
    # the cloud outside it; widen the limits instead and keep the full canvas.
    if self.side and self.stretch.get():self.ax.set_aspect('auto')
    else:self.ax.set_aspect('equal',adjustable='datalim')
    if limits:self.ax.set_xlim(limits[0]);self.ax.set_ylim(limits[1])
    else:
     for axis,k in (('x',h),('y',v)):
      lo,hi=fit[:,k].min(),fit[:,k].max();pad=max((hi-lo)*.05,1) if self.side else 20
      getattr(self.ax,'set_'+axis+'lim')(lo-pad,hi+pad)
   self.lasso=LassoSelector(self.ax,self.select,button=1,props=dict(color='white',linewidth=1.5))
   # Belt and braces with the covering rectangle above: never stop painting at
   # the axes edge, which used to cut the cloud off inside the canvas.
   for artist in (self.full_artist,self.lod_artist):
    if artist is not None:artist.set_clip_on(False)
   if self.show_numbers.get():
    for lab,center in self.label_centers().items():
     if lab not in self.labels[ix]:continue
     if lab==1 and not self.show_pallet.get():continue
     kw=dict(fontsize=10,color='white',weight='bold',ha='center',bbox=dict(facecolor='#0a1322',alpha=.85,edgecolor='none',pad=2))
     if self.view3d:self.ax.text(*center,str(lab),**kw)
     else:
      h,v=self.plane()
      z=float(np.median(self.xyz[self.labels==lab,2])) if self.side else center[v]
      self.ax.text(center[h],z,str(lab),**kw)
   # On the figure, not the axes: a 3D axes sits inset from the canvas edge.
   caption=f'{self.current.name}  |  {len(ix):,} / {len(self.labels):,} points'
   if focus is not None:caption+=f'  |  focus label {focus}'
   if self.side:caption+=('  |  front XZ' if self.direction=='앞' else '  |  side YZ')+(' (height stretched)' if self.stretch.get() else '')
   if self.clip_enabled.get():caption+='  |  clipped'
   if not len(ix):self.fig.text(.5,.5,'이 구간에 표시할 점이 없습니다. 구간을 넓히세요.',ha='center',color='white')
   self.fig.text(.012,.975,caption,color='#c6d8ed',fontsize=9,va='top',
                 bbox=dict(facecolor=BACKGROUND,alpha=.75,edgecolor='none',pad=3))
  self.reset_view=False
  # Paint the sparse pass first so a toggle feels instant, then fill in full detail.
  self.show_moving()
  if self.lod_artist is not None:self.settle_later()
 def screen_xy(self):
  """Every point in the flat coordinate space the lasso reports its vertices in."""
  if not self.view3d:return self.xyz[:,self.plane()]
  xs,ys,_=proj3d.proj_transform(self.xyz[:,0],self.xyz[:,1],self.xyz[:,2],self.ax.get_proj())
  return np.column_stack([xs,ys])
 def viewer_depth(self):
  """How near the viewer each point is, larger being nearer."""
  if self.view3d:return proj3d.proj_transform(self.xyz[:,0],self.xyz[:,1],self.xyz[:,2],self.ax.get_proj())[2]
  # A flat view drops one axis, and the viewer stands on its near side.
  return {(0,1):self.xyz[:,2],(0,2):-self.xyz[:,1],(1,2):-self.xyz[:,0]}[self.plane()]
 def select(self,vertices):
  if self.busy or self.labels is None:return
  vertices=np.asarray(vertices,dtype=float)
  if len(vertices)==0:return
  # A click lands here as a lasso a few pixels wide; treat it as picking one object.
  if len(vertices)<3 or np.ptp(self.ax.transData.transform(vertices),axis=0).max()<CLICK_SLOP:
   self.pick(vertices[0]);return
  inside=Polygon(vertices).contains_points(self.screen_xy())
  inside&=self.visible_mask()
  focus=self.focused_label()
  if focus is not None:inside&=self.labels==focus
  if not self.show_pallet.get():inside&=self.labels!=1
  self.pick_highlight=False
  self.choose(inside,f'라벨 {focus} 안의 영역' if focus is not None else '전체 영역',restrict_focus=True)
 def pick(self,point):
  pixels=self.ax.transData.transform(self.screen_xy())
  gap=((pixels-self.ax.transData.transform(point))**2).sum(1)
  gap[~self.visible_mask()]=np.inf
  if not self.show_pallet.get():gap[self.labels==1]=np.inf
  near=int(np.argmin(gap))
  if gap[near]>PICK_RADIUS**2:
   if self.side:
    self.selected=None;self.pick_highlight=False;self.status.set('점 선택을 해제했습니다. 옆에서 보기와 부재 고정은 유지됩니다.');self.draw()
   else:self.clear_selection()
   return
  # Of everything under the cursor take the nearest to the viewer, so a click
  # lands on what it looks like it is on and not on something hidden behind.
  near=int(np.argmax(np.where(gap<=PICK_RADIUS**2,self.viewer_depth(),-np.inf)))
  lab=int(self.labels[near])
  if lab==1 and not self.show_pallet.get():self.clear_selection();return
  whole=(self.labels==lab)&self.visible_mask()
  # Clicking the object that is already selected on its own clears it again.
  if self.selected is not None and not self.press_key and np.array_equal(self.selected,whole):self.clear_selection();return
  if self.focus!=lab:self.reset_clip(False)
  self.focus=lab;self.pick_highlight=True
  self.choose(whole,f'라벨 {lab} 전체 (이후 드래그는 이 부재 안에서만)')
 def choose(self,mask,what='영역',restrict_focus=False):
  key=self.press_key or ''
  if self.selected is not None and 'shift' in key:mask=self.selected|mask
  elif self.selected is not None and 'alt' in key:mask=self.selected&~mask
  if restrict_focus and self.focused_label() is not None:mask&=self.labels==self.focus
  if restrict_focus and not self.show_pallet.get():mask&=self.labels!=1
  mask&=self.visible_mask()
  count=int(mask.sum())
  self.selected=mask if count else None
  self.status.set(f'{what} {count:,}점 선택됨. 라벨 번호를 입력하고 적용하세요.' if count else '선택한 점이 없습니다.')
  self.draw()
 def assign(self):
  if self.busy:return
  if self.selected is None or not np.any(self.selected):return
  wanted=int(self.selected.sum());self.selected&=self.visible_mask();count=int(self.selected.sum())
  if not count:
   self.selected=None;self.pick_highlight=False
   self.status.set('선택한 점이 모두 숨겨져 있어 적용하지 않았습니다. 구간을 넓히거나 팔레트 표시를 켜세요.');self.draw();return
  try:
   value=int(self.label.get())
   if value<1 or value>100000:raise ValueError()
  except ValueError:messagebox.showerror('라벨 번호','1 이상의 정수를 입력하세요.');return
  self.undo.append((np.flatnonzero(self.selected),self.labels[self.selected].copy(),self.review[self.selected].copy()))
  if len(self.undo)>20:self.undo.pop(0)
  self.labels[self.selected]=value;self.review[self.selected]=0;self.selected=None;self.pick_highlight=False;self.labels_version+=1;self.draw()
  scope=f' 드래그는 라벨 {self.focus} 안으로 제한됩니다.' if self.focused_label() is not None else ''
  hidden=f' 숨겨진 {wanted-count:,}점은 빼고 {count:,}점에 적용했습니다.' if wanted>count else ''
  self.status.set('수정했습니다. 결과 저장 버튼으로 새 PLY를 저장하세요.'+hidden+scope)
 def new_object(self):
  if self.labels is not None:self.label.set(str(int(self.labels.max())+1));self.assign()
 def rollback(self):
  if self.busy or not self.undo:return
  ix,old,review=self.undo.pop();self.labels[ix]=old;self.review[ix]=review;self.selected=None;self.pick_highlight=False;self.labels_version+=1;self.draw()
 def toggle(self):self.set_direction('2D' if self.view3d else '3D')
 def is_source(self,path):
  """True when path is one of the PLY files we read, which must never be written over."""
  for other in [self.current,*self.files]:
   try:
    if other is not None and path.resolve()==Path(other).resolve():return True
   except OSError:pass
  return False
 def save(self):
  if self.labels is None or self.busy:return
  dest=filedialog.asksaveasfilename(title='수정 결과 저장',initialdir=self.out.get(),
                                    initialfile=f'{self.current.stem}_labeled.ply',
                                    defaultextension='.ply',filetypes=[('Point cloud','*.ply')])
  if not dest:return
  dest=Path(dest)
  if self.is_source(dest):
   messagebox.showerror('저장 위치','불러온 원본 PLY는 덮어쓸 수 없습니다. 다른 이름으로 저장하세요.');return
  try:
   # Preserve pallet=1; compact object IDs after manual merges/deletions.
   out=np.ones(len(self.labels),np.int32)
   for new,old in enumerate(np.unique(self.labels[self.labels>1]),2):out[self.labels==old]=new
   # The file dialog already asked before replacing anything.
   save_cloud(self.ply,out,dest,self.review,overwrite=True)
   if self.focused_label() is not None:self.focus=int(out[np.flatnonzero(self.labels==self.focus)[0]])
   self.selected=None;self.pick_highlight=False
   self.labels=out;self.undo=[];self.labels_version+=1;self.draw();self.status.set(f'저장 완료 · {dest}');messagebox.showinfo('저장 완료',f'{dest}\n\nCloudCompare에서 instance_label을 선택해 확인하세요.')
  except Exception as e:messagebox.showerror('저장 오류',str(e))

if __name__=='__main__':
 import sys
 win=tk.Tk();app=App(win)
 if len(sys.argv)>1:app.add(sys.argv[1:]);app.open_selected()
 win.mainloop()
