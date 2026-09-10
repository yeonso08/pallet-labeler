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
from engine import read_cloud,infer,save_cloud,LABEL
from scipy.spatial import cKDTree
from viewer import BACKGROUND,display_indices,point_colors

ROOT=Path(__file__).resolve().parent

class App:
 def __init__(self,win):
  self.win=win;win.title('Pallet Labeler v2.1 · 컬러 점군 보기');win.geometry('1250x850')
  self.jobs=queue.Queue();self.busy=False;self.ply=None;self.labels=None;self.selected=None;self.current=None;self.model=None
  self.files=[];self.undo=[];self.view3d=True;self.reset_view=True
  config_path=ROOT/'model_config.json'
  self.model_options=json.loads(config_path.read_text()) if config_path.exists() else {'기본 모델':{'file':'model.joblib','split':.1}}
  self.active_model_name=next(iter(self.model_options));self.active_model_file=ROOT/self.model_options[self.active_model_name]['file']
  style=ttk.Style();style.theme_use('clam');style.configure('TButton',padding=7);style.configure('TLabel',padding=3)
  head=ttk.Frame(win,padding=(18,12));head.pack(fill='x')
  ttk.Label(head,text='PALLET LABELER',font=('',20,'bold')).pack(side='left')
  ttk.Label(head,text='팔레트 1  ·  부자재 2, 3, 4…  ·  자동 결과는 확인 후 저장').pack(side='left',padx=20)
  body=ttk.Panedwindow(win,orient='horizontal');body.pack(fill='both',expand=True,padx=16)
  left=ttk.Frame(body,width=265,padding=8);right=ttk.Frame(body);body.add(left,weight=0);body.add(right,weight=1)
  ttk.Label(left,text='자동 라벨링 모델').pack(anchor='w')
  self.model_choice=tk.StringVar(value=self.active_model_name)
  self.model_combo=ttk.Combobox(left,textvariable=self.model_choice,values=list(self.model_options),state='readonly',width=25);self.model_combo.pack(fill='x',pady=(0,6))
  self.model_combo.bind('<<ComboboxSelected>>',self.change_model)
  ttk.Button(left,text='PLY 파일 추가',command=self.add_files).pack(fill='x')
  ttk.Button(left,text='폴더에서 추가',command=self.add_folder).pack(fill='x',pady=4)
  self.listbox=tk.Listbox(left,width=27,height=10,exportselection=False);self.listbox.pack(fill='x',pady=8)
  self.listbox.bind('<Double-1>',lambda _:self.open_selected())
  ttk.Button(left,text='선택 파일 열기 / 자동 라벨링',command=self.open_selected).pack(fill='x')
  ttk.Label(left,text='물체 경계 분리 기준').pack(anchor='w',pady=(12,0))
  self.split=tk.DoubleVar(value=self.model_options[self.active_model_name]['split']);ttk.Scale(left,from_=.02,to=.35,variable=self.split).pack(fill='x')
  ttk.Label(left,text='왼쪽: 더 잘게 분리 / 오른쪽: 더 크게 합침',font=('',10)).pack(anchor='w')
  self.out=tk.StringVar(value=str(ROOT/'results'))
  ttk.Button(left,text='선택 파일 다시 자동 라벨링',command=lambda:self.open_selected(force=True)).pack(fill='x',pady=4)
  ttk.Label(left,text='저장 폴더').pack(anchor='w',pady=(12,0))
  ttk.Entry(left,textvariable=self.out,width=27).pack(fill='x')
  ttk.Button(left,text='저장 폴더 선택',command=self.choose_out).pack(fill='x',pady=4)
  ttk.Button(left,text='추가한 파일 일괄 처리',command=self.batch).pack(fill='x',pady=6)
  ttk.Label(left,text='일괄 결과는 자동 초안입니다.\nCloudCompare에서 확인하세요.\n원본 파일은 변경하지 않습니다.',wraplength=245).pack(anchor='w',pady=8)
  controls=ttk.Frame(right);controls.pack(fill='x')
  self.label=tk.StringVar(value='2')
  ttk.Label(controls,text='선택한 점 → 라벨').pack(side='left');ttk.Entry(controls,textvariable=self.label,width=5).pack(side='left')
  ttk.Button(controls,text='적용',command=self.assign).pack(side='left',padx=3)
  ttk.Button(controls,text='새 물체로 분리',command=self.new_object).pack(side='left')
  ttk.Button(controls,text='되돌리기',command=self.rollback).pack(side='left',padx=3)
  ttk.Button(controls,text='2D / 3D',command=self.toggle).pack(side='left')
  ttk.Button(controls,text='수정 결과 저장',command=self.save).pack(side='right')
  display=ttk.Frame(right);display.pack(fill='x',pady=5)
  self.color_mode=tk.StringVar(value='원본 색상');self.density=tk.StringVar(value='촘촘하게');self.point_size=tk.DoubleVar(value=2.0);self.show_pallet=tk.BooleanVar(value=True);self.show_numbers=tk.BooleanVar(value=True)
  self.color_combo=ttk.Combobox(display,textvariable=self.color_mode,values=['물체별 색상','원본 색상','높이 색상'],state='readonly',width=12);self.color_combo.pack(side='left');self.color_combo.bind('<<ComboboxSelected>>',lambda _:self.draw())
  self.density_combo=ttk.Combobox(display,textvariable=self.density,values=['빠르게','촘촘하게','전체 점'],state='readonly',width=10);self.density_combo.pack(side='left',padx=5);self.density_combo.bind('<<ComboboxSelected>>',lambda _:self.draw())
  ttk.Checkbutton(display,text='팔레트 표시',variable=self.show_pallet,command=self.draw).pack(side='left')
  ttk.Checkbutton(display,text='번호 표시',variable=self.show_numbers,command=self.draw).pack(side='left')
  ttk.Label(display,text='점 크기').pack(side='left');size=ttk.Scale(display,from_=.5,to=5,variable=self.point_size,length=75);size.pack(side='left');size.bind('<ButtonRelease-1>',lambda _:self.draw())
  ttk.Button(display,text='화면 맞춤',command=self.fit_view).pack(side='right')
  ttk.Button(display,text='선택 해제',command=self.clear_selection).pack(side='right',padx=4)
  ttk.Label(right,text='3D: 드래그로 회전 · 휠로 확대  |  2D: 영역 선택 / 오른쪽 클릭으로 물체 선택  |  선택한 점은 흰색').pack(anchor='w')
  self.fig=Figure(figsize=(8,6),facecolor=BACKGROUND);self.ax=self.fig.add_subplot(111)
  self.canvas=FigureCanvasTkAgg(self.fig,master=right);self.canvas.get_tk_widget().pack(fill='both',expand=True)
  self.toolbar=NavigationToolbar2Tk(self.canvas,right);self.lasso=None
  self.canvas.mpl_connect('button_press_event',self.pick_instance)
  self.canvas.mpl_connect('scroll_event',self.zoom)
  self.status=tk.StringVar(value='PLY 파일을 추가하고 자동 라벨링을 시작하세요. 제공된 스캐너 좌표와 단위를 사용합니다.')
  ttk.Label(win,textvariable=self.status,padding=12,wraplength=1200).pack(fill='x')
  win.after(100,self.poll);self.draw()
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
     self.current,self.ply,self.xyz,self.labels,self.review,self.info=data;self.selected=None;self.undo=[];self.xy_tree=cKDTree(self.xyz[:,:2]);self.reset_view=True
     self.status.set(f'{self.current.name} · {len(self.labels):,}점 · 부자재 {self.info["objects"]}개 추정 · 경계/불확실 점 {self.info["review_fraction"]:.1%} · 자동 초안');self.draw()
  except queue.Empty:pass
  self.win.after(100,self.poll)
 def fit_view(self):self.reset_view=True;self.draw()
 def clear_selection(self):self.selected=None;self.draw()
 def zoom(self,event):
  if event.inaxes!=self.ax or self.labels is None:return
  factor=.85 if event.button=='up' else 1/.85
  axes=['x','y','z'] if self.view3d else ['x','y']
  for axis in axes:
   lo,hi=getattr(self.ax,'get_'+axis+'lim')();center=(lo+hi)/2;half=(hi-lo)*factor/2
   getattr(self.ax,'set_'+axis+'lim')(center-half,center+half)
  self.canvas.draw_idle()
 def draw(self):
  camera=None;limits=None
  if not self.reset_view and self.labels is not None:
   if self.view3d and getattr(self.ax,'name','')=='3d':camera=(self.ax.elev,self.ax.azim,self.ax.roll);limits=(self.ax.get_xlim(),self.ax.get_ylim(),self.ax.get_zlim())
   elif not self.view3d and getattr(self.ax,'name','')!='3d':limits=(self.ax.get_xlim(),self.ax.get_ylim())
  if self.lasso:self.lasso.disconnect_events();self.lasso=None
  self.fig.clear();self.fig.set_facecolor(BACKGROUND);self.ax=self.fig.add_axes([.01,.01,.98,.98],projection='3d' if self.view3d else None)
  self.ax.set_facecolor(BACKGROUND);self.ax.set_axis_off()
  if self.labels is None:
   if self.view3d:self.ax.text2D(.5,.5,'Open a PLY file to begin',ha='center',color='white',transform=self.ax.transAxes)
   else:self.ax.text(.5,.5,'Open a PLY file to begin',ha='center',color='white',transform=self.ax.transAxes)
  else:
   budget={'빠르게':60000,'촘촘하게':200000,'전체 점':None}[self.density.get()]
   ix=display_indices(len(self.labels),budget)
   if not self.show_pallet.get():ix=ix[self.labels[ix]!=1]
   pts=self.xyz[ix];colors=point_colors(self.ply,self.xyz,self.labels,ix,self.color_mode.get(),self.selected)
   size=float(self.point_size.get())
   if self.view3d:
    self.ax.set_proj_type('ortho')
    self.ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=colors,s=size,depthshade=False,linewidths=0,antialiased=False)
    extent=np.maximum(np.ptp(self.xyz,axis=0),1);self.ax.set_box_aspect(extent,zoom=1.12)
    self.ax.view_init(*(camera or (55,-65,0)))
    if limits:
     self.ax.set_xlim(limits[0]);self.ax.set_ylim(limits[1]);self.ax.set_zlim(limits[2])
    else:
     for k,axis in enumerate('xyz'):
      lo,hi=np.quantile(self.xyz[:,k],[0,1]);pad=max((hi-lo)*.025,.5);getattr(self.ax,'set_'+axis+'lim')(lo-pad,hi+pad)
   else:
    self.ax.scatter(pts[:,0],pts[:,1],c=colors,s=size,linewidths=0,antialiased=False);self.ax.set_aspect('equal');self.lasso=LassoSelector(self.ax,self.select,button=1,props=dict(color='white',linewidth=1.5))
    if limits:self.ax.set_xlim(limits[0]);self.ax.set_ylim(limits[1])
    else:self.ax.set_xlim(self.xyz[:,0].min()-20,self.xyz[:,0].max()+20);self.ax.set_ylim(self.xyz[:,1].min()-20,self.xyz[:,1].max()+20)
   if self.show_numbers.get():
    for lab in np.unique(self.labels):
     if lab==1 and not self.show_pallet.get():continue
     v=self.xyz[self.labels==lab];center=np.median(v,axis=0);center[2]=np.quantile(v[:,2],.9)+3
     kw=dict(fontsize=10,color='white',weight='bold',ha='center',bbox=dict(facecolor='#0a1322',alpha=.85,edgecolor='none',pad=2))
     if self.view3d:self.ax.text(*center,str(lab),**kw)
     else:self.ax.text(*center[:2],str(lab),**kw)
   caption=f'{self.current.name}  |  {len(ix):,} / {len(self.labels):,} points'
   if self.view3d:self.ax.text2D(.015,.97,caption,color='#c6d8ed',fontsize=9,transform=self.ax.transAxes)
   else:self.ax.text(.015,.97,caption,color='#c6d8ed',fontsize=9,transform=self.ax.transAxes)
  self.reset_view=False;self.canvas.draw_idle()
 def select(self,vertices):
  if self.busy or len(vertices)<3:return
  self.selected=Polygon(vertices).contains_points(self.xyz[:,:2]);
  if not self.show_pallet.get():self.selected &= self.labels!=1
  self.status.set(f'{self.selected.sum():,}점 선택됨. 적용할 라벨 번호를 입력하세요.');self.draw()
 def pick_instance(self,event):
  if self.busy or self.labels is None or self.view3d or event.button!=3 or event.inaxes!=self.ax or self.toolbar.mode:return
  d,ix=self.xy_tree.query([event.xdata,event.ydata])
  if d>20:return
  lab=self.labels[ix]
  if lab==1 and not self.show_pallet.get():return
  self.selected=self.labels==lab;self.status.set(f'라벨 {lab} 전체 {self.selected.sum():,}점 선택됨. 합칠 대상 번호를 입력하고 적용하세요.');self.draw()
 def assign(self):
  if self.busy:return
  if self.selected is None or not np.any(self.selected):return
  try:
   value=int(self.label.get())
   if value<1 or value>100000:raise ValueError()
  except ValueError:messagebox.showerror('라벨 번호','1 이상의 정수를 입력하세요.');return
  self.undo.append((np.flatnonzero(self.selected),self.labels[self.selected].copy(),self.review[self.selected].copy()))
  if len(self.undo)>20:self.undo.pop(0)
  self.labels[self.selected]=value;self.review[self.selected]=0;self.selected=None;self.draw();self.status.set('수정했습니다. 결과 저장 버튼으로 새 PLY를 저장하세요.')
 def new_object(self):
  if self.labels is not None:self.label.set(str(int(self.labels.max())+1));self.assign()
 def rollback(self):
  if self.busy or not self.undo:return
  ix,old,review=self.undo.pop();self.labels[ix]=old;self.review[ix]=review;self.selected=None;self.draw()
 def toggle(self):self.view3d=not self.view3d;self.reset_view=True;self.draw()
 def save(self):
  if self.labels is None or self.busy:return
  try:
   # Preserve pallet=1; compact object IDs after manual merges/deletions.
   out=np.ones(len(self.labels),np.int32)
   for new,old in enumerate(np.unique(self.labels[self.labels>1]),2):out[self.labels==old]=new
   dest=self.unique_path(self.out.get(),self.current.stem);save_cloud(self.ply,out,dest,self.review)
   self.labels=out;self.undo=[];self.draw();self.status.set(f'저장 완료 · {dest}');messagebox.showinfo('저장 완료',f'{dest}\n\nCloudCompare에서 instance_label을 선택해 확인하세요.')
  except Exception as e:messagebox.showerror('저장 오류',str(e))

if __name__=='__main__':
 import sys
 win=tk.Tk();app=App(win)
 if len(sys.argv)>1:app.add(sys.argv[1:]);app.open_selected()
 win.mainloop()
