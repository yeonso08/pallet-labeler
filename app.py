"""Desktop interface: local prediction, point selection, correction and safe export."""
from pathlib import Path
import json,queue,threading,time,traceback
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
  self.full_artist=None;self.lod_artist=None;self.moving=False;self.settle_timer=None;self.labels_version=0;self.centers=None
  self.press_key=None;self.pan_from=None
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
  ttk.Entry(controls,textvariable=self.label,width=5).pack(side='left')
  ttk.Button(controls,text='적용',command=self.assign).pack(side='left',padx=3)
  ttk.Button(controls,text='새 물체로 분리',command=self.new_object).pack(side='left')
  ttk.Button(controls,text='되돌리기',command=self.rollback).pack(side='left',padx=3)
  ttk.Button(controls,text='2D / 3D',command=self.toggle).pack(side='left')
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
  ttk.Label(right,text='왼쪽 드래그: 영역 선택  ·  왼쪽 클릭: 물체 선택  ·  Shift: 추가, Alt: 빼기  ·  오른쪽 드래그: 3D 회전 / 2D 이동  ·  휠 누르고 드래그: 이동  ·  Esc: 해제',style='Muted.TLabel').pack(anchor='w')
  self.fig=Figure(figsize=(8,6),facecolor=BACKGROUND);self.ax=self.fig.add_subplot(111)
  self.canvas=FigureCanvasTkAgg(self.fig,master=right);self.canvas.get_tk_widget().pack(fill='both',expand=True)
  self.toolbar=Toolbar(self.canvas,right);self.lasso=None
  theme.style_toolbar(self.toolbar,fonts)
  self.canvas.mpl_connect('scroll_event',self.zoom)
  self.canvas.mpl_connect('button_press_event',self.start_move)
  self.canvas.mpl_connect('motion_notify_event',self.pan_move)
  self.canvas.mpl_connect('button_release_event',self.end_move)
  win.bind('<Escape>',lambda _:self.clear_selection())
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
     self.status.set(f'{self.current.name} · {len(self.labels):,}점 · 부자재 {self.info["objects"]}개 추정 · 경계/불확실 점 {self.info["review_fraction"]:.1%} · 자동 초안');self.draw()
  except queue.Empty:pass
  self.win.after(100,self.poll)
 def fit_view(self):self.reset_view=True;self.draw()
 def clear_selection(self):
  if self.selected is None:return
  self.selected=None;self.status.set('선택을 해제했습니다.');self.draw()
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
  else:self.lod_artist=self.ax.scatter(p[:,0],p[:,1],c=c,s=size,linewidths=0,antialiased=False)
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
  camera=None;limits=None
  if not self.reset_view and self.labels is not None:
   if self.view3d and getattr(self.ax,'name','')=='3d':camera=(self.ax.elev,self.ax.azim,self.ax.roll);limits=(self.ax.get_xlim(),self.ax.get_ylim(),self.ax.get_zlim())
   elif not self.view3d and getattr(self.ax,'name','')!='3d':limits=(self.ax.get_xlim(),self.ax.get_ylim())
  if self.lasso:self.lasso.disconnect_events();self.lasso=None
  self.cancel_settle();self.moving=False;self.full_artist=None;self.lod_artist=None
  self.fig.clear();self.fig.set_facecolor(BACKGROUND);self.ax=self.fig.add_axes([.01,.01,.98,.98],projection='3d' if self.view3d else None)
  self.ax.set_facecolor(BACKGROUND);self.ax.set_axis_off()
  if self.labels is None:
   self.fig.text(.5,.5,'왼쪽에서 PLY 파일을 추가한 뒤 열어 주세요',ha='center',va='center',color=theme.MUTED,fontsize=11)
  else:
   budget={'빠르게':60000,'촘촘하게':200000,'전체 점':None}[self.density.get()]
   ix=display_indices(len(self.labels),budget)
   if not self.show_pallet.get():ix=ix[self.labels[ix]!=1]
   pts=self.xyz[ix];colors=point_colors(self.ply,self.xyz,self.labels,ix,self.color_mode.get(),self.selected)
   size=float(self.point_size.get())
   if self.view3d:
    self.ax.set_proj_type('ortho')
    # The left button draws the lasso, so rotating moves to the right button.
    self.ax.mouse_init(rotate_btn=3,pan_btn=2,zoom_btn=[])
    self.full_artist=self.ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=colors,s=size,depthshade=False,linewidths=0,antialiased=False)
    self.add_lod(pts,colors,size)
    extent=np.maximum(np.ptp(self.xyz,axis=0),1);self.ax.set_box_aspect(extent,zoom=1.12)
    self.ax.view_init(*(camera or (55,-65,0)))
    if limits:
     self.ax.set_xlim(limits[0]);self.ax.set_ylim(limits[1]);self.ax.set_zlim(limits[2])
    else:
     for k,axis in enumerate('xyz'):
      lo,hi=np.quantile(self.xyz[:,k],[0,1]);pad=max((hi-lo)*.025,.5);getattr(self.ax,'set_'+axis+'lim')(lo-pad,hi+pad)
   else:
    self.full_artist=self.ax.scatter(pts[:,0],pts[:,1],c=colors,s=size,linewidths=0,antialiased=False);self.add_lod(pts,colors,size)
    self.ax.set_aspect('equal')
    if limits:self.ax.set_xlim(limits[0]);self.ax.set_ylim(limits[1])
    else:self.ax.set_xlim(self.xyz[:,0].min()-20,self.xyz[:,0].max()+20);self.ax.set_ylim(self.xyz[:,1].min()-20,self.xyz[:,1].max()+20)
   self.lasso=LassoSelector(self.ax,self.select,button=1,props=dict(color='white',linewidth=1.5))
   # mplot3d forces its axes region square (Axes3D.apply_aspect), which on a wide canvas
   # leaves wide empty margins and cuts the cloud off well inside them. Paint over the
   # whole figure instead of stopping at the axes rectangle.
   for artist in (self.full_artist,self.lod_artist):
    if artist is not None:artist.set_clip_on(False)
   if self.show_numbers.get():
    for lab,center in self.label_centers().items():
     if lab==1 and not self.show_pallet.get():continue
     kw=dict(fontsize=10,color='white',weight='bold',ha='center',bbox=dict(facecolor='#0a1322',alpha=.85,edgecolor='none',pad=2))
     if self.view3d:self.ax.text(*center,str(lab),**kw)
     else:self.ax.text(*center[:2],str(lab),**kw)
   # On the figure, not the axes: a 3D axes sits inset from the canvas edge.
   self.fig.text(.012,.975,f'{self.current.name}  |  {len(ix):,} / {len(self.labels):,} points',color='#c6d8ed',fontsize=9,va='top',
                 bbox=dict(facecolor=BACKGROUND,alpha=.75,edgecolor='none',pad=3))
  self.reset_view=False
  # Paint the sparse pass first so a toggle feels instant, then fill in full detail.
  self.show_moving()
  if self.lod_artist is not None:self.settle_later()
 def screen_xy(self):
  """Every point in the flat coordinate space the lasso reports its vertices in."""
  if not self.view3d:return self.xyz[:,:2]
  xs,ys,_=proj3d.proj_transform(self.xyz[:,0],self.xyz[:,1],self.xyz[:,2],self.ax.get_proj())
  return np.column_stack([xs,ys])
 def select(self,vertices):
  if self.busy or self.labels is None:return
  vertices=np.asarray(vertices,dtype=float)
  # A click lands here as a lasso a few pixels wide; treat it as picking one object.
  if len(vertices)<3 or np.ptp(self.ax.transData.transform(vertices),axis=0).max()<CLICK_SLOP:
   self.pick(vertices[0]);return
  inside=Polygon(vertices).contains_points(self.screen_xy())
  if not self.show_pallet.get():inside&=self.labels!=1
  self.choose(inside)
 def pick(self,point):
  pixels=self.ax.transData.transform(self.screen_xy())
  gap=((pixels-self.ax.transData.transform(point))**2).sum(1)
  near=int(np.argmin(gap))
  if gap[near]>PICK_RADIUS**2:self.clear_selection();return
  lab=int(self.labels[near])
  if lab==1 and not self.show_pallet.get():self.clear_selection();return
  whole=self.labels==lab
  # Clicking the object that is already selected on its own clears it again.
  if self.selected is not None and not self.press_key and np.array_equal(self.selected,whole):self.clear_selection();return
  self.choose(whole,f'라벨 {lab} 전체')
 def choose(self,mask,what='영역'):
  key=self.press_key or ''
  if self.selected is not None and 'shift' in key:mask=self.selected|mask
  elif self.selected is not None and 'alt' in key:mask=self.selected&~mask
  count=int(mask.sum())
  self.selected=mask if count else None
  self.status.set(f'{what} {count:,}점 선택됨. 라벨 번호를 입력하고 적용하세요.' if count else '선택한 점이 없습니다.')
  self.draw()
 def assign(self):
  if self.busy:return
  if self.selected is None or not np.any(self.selected):return
  try:
   value=int(self.label.get())
   if value<1 or value>100000:raise ValueError()
  except ValueError:messagebox.showerror('라벨 번호','1 이상의 정수를 입력하세요.');return
  self.undo.append((np.flatnonzero(self.selected),self.labels[self.selected].copy(),self.review[self.selected].copy()))
  if len(self.undo)>20:self.undo.pop(0)
  self.labels[self.selected]=value;self.review[self.selected]=0;self.selected=None;self.labels_version+=1;self.draw();self.status.set('수정했습니다. 결과 저장 버튼으로 새 PLY를 저장하세요.')
 def new_object(self):
  if self.labels is not None:self.label.set(str(int(self.labels.max())+1));self.assign()
 def rollback(self):
  if self.busy or not self.undo:return
  ix,old,review=self.undo.pop();self.labels[ix]=old;self.review[ix]=review;self.selected=None;self.labels_version+=1;self.draw()
 def toggle(self):self.view3d=not self.view3d;self.reset_view=True;self.draw()
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
   self.labels=out;self.undo=[];self.labels_version+=1;self.draw();self.status.set(f'저장 완료 · {dest}');messagebox.showinfo('저장 완료',f'{dest}\n\nCloudCompare에서 instance_label을 선택해 확인하세요.')
  except Exception as e:messagebox.showerror('저장 오류',str(e))

if __name__=='__main__':
 import sys
 win=tk.Tk();app=App(win)
 if len(sys.argv)>1:app.add(sys.argv[1:]);app.open_selected()
 win.mainloop()
