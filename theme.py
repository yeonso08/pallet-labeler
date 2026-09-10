"""Dark instrument styling for the Tk chrome, keyed to the point cloud's own colours.

The panels sit a shade above the scan so the canvas reads as a recessed well, and
amber - the colour of the pallet timber in the scans - is reserved for the primary
action and for controls that are currently switched on.
"""
import tkinter as tk
from tkinter import ttk, font as tkfont
from viewer import BACKGROUND

SCAN=BACKGROUND
WELL='#0b1320'
PANEL='#172233'
CONTROL='#22304a'
HOVER='#2c3d59'
BORDER='#31425c'
TEXT='#e6edf7'
MUTED='#8496b0'
ACCENT='#f0a92e'
ACCENT_DIM='#c9871f'
ON_ACCENT='#1a1204'
SELECT='#1f3a5c'
HANDLE='#b9c8dd'

# Korean UI faces first: the whole interface is Korean and the fallbacks render it poorly.
FACES=['Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR','Segoe UI','Helvetica Neue']

def _family(root):
    available={f.lower() for f in tkfont.families(root)}
    for name in FACES:
        if name.lower() in available:return name
    return ''

def apply(root):
    """Style every ttk class this app uses and return the font set."""
    family=_family(root)
    fonts=dict(wordmark=(family,16,'bold'),ui=(family,12),small=(family,11))
    root.configure(background=PANEL)
    root.option_add('*TCombobox*Listbox.background',CONTROL)
    root.option_add('*TCombobox*Listbox.foreground',TEXT)
    root.option_add('*TCombobox*Listbox.selectBackground',SELECT)
    root.option_add('*TCombobox*Listbox.selectForeground',TEXT)
    root.option_add('*TCombobox*Listbox.font',fonts['ui'])

    s=ttk.Style();s.theme_use('clam')
    # clam's own defaults for these are light greys; anything not restyled below
    # inherits them and shows up as a bright outline on the dark ground.
    s.configure('.',background=PANEL,foreground=TEXT,fieldbackground=CONTROL,font=fonts['ui'],
                bordercolor=BORDER,lightcolor=PANEL,darkcolor=PANEL,troughcolor=WELL,arrowcolor=MUTED,
                insertcolor=TEXT,selectbackground=SELECT,selectforeground=TEXT,focuscolor=PANEL)
    s.configure('TFrame',background=PANEL)

    s.configure('TLabel',background=PANEL,foreground=TEXT,font=fonts['ui'])
    s.configure('Wordmark.TLabel',font=fonts['wordmark'])
    s.configure('Muted.TLabel',foreground=MUTED,font=fonts['small'])

    # clam draws a bevel from lightcolor/darkcolor; matching them to the fill keeps buttons flat.
    # clam ships TButton with width=-11, a minimum of 11 character widths, which makes
    # every button the same slab regardless of its label; 0 lets them size to their text.
    s.configure('TButton',background=CONTROL,foreground=TEXT,bordercolor=BORDER,lightcolor=CONTROL,darkcolor=CONTROL,focuscolor=PANEL,relief='flat',anchor='center',width=0,padding=(12,8),font=fonts['ui'])
    s.map('TButton',background=[('pressed',BORDER),('active',HOVER),('disabled',PANEL)],foreground=[('disabled',MUTED)],lightcolor=[('pressed',BORDER),('active',HOVER)],darkcolor=[('pressed',BORDER),('active',HOVER)],bordercolor=[('disabled',PANEL)])

    s.configure('Accent.TButton',background=ACCENT,foreground=ON_ACCENT,bordercolor=ACCENT,lightcolor=ACCENT,darkcolor=ACCENT)
    s.map('Accent.TButton',background=[('pressed',ACCENT_DIM),('active','#ffbe4a'),('disabled',CONTROL)],foreground=[('disabled',MUTED)],lightcolor=[('pressed',ACCENT_DIM),('active','#ffbe4a')],darkcolor=[('pressed',ACCENT_DIM),('active','#ffbe4a')],bordercolor=[('disabled',BORDER)])

    s.configure('TEntry',fieldbackground=CONTROL,foreground=TEXT,insertcolor=TEXT,bordercolor=BORDER,lightcolor=CONTROL,darkcolor=CONTROL,selectbackground=SELECT,selectforeground=TEXT,padding=(8,6))
    s.map('TEntry',bordercolor=[('focus',ACCENT)],lightcolor=[('focus',ACCENT)],darkcolor=[('focus',ACCENT)])

    s.configure('TCombobox',fieldbackground=CONTROL,background=CONTROL,foreground=TEXT,arrowcolor=MUTED,bordercolor=BORDER,lightcolor=CONTROL,darkcolor=CONTROL,selectbackground=CONTROL,selectforeground=TEXT,padding=(8,6))
    s.map('TCombobox',fieldbackground=[('readonly',CONTROL),('disabled',PANEL)],foreground=[('disabled',MUTED)],arrowcolor=[('active',TEXT),('disabled',BORDER)],bordercolor=[('focus',ACCENT)],lightcolor=[('focus',ACCENT)],darkcolor=[('focus',ACCENT)])

    # Display switches: lit by amber lettering on a raised face, so the only solid
    # amber on screen stays the save action.
    s.configure('Off.TButton',background=PANEL,foreground=MUTED,bordercolor=BORDER,lightcolor=PANEL,darkcolor=PANEL,padding=(10,7))
    s.map('Off.TButton',background=[('pressed',CONTROL),('active',CONTROL)],foreground=[('active',TEXT)],lightcolor=[('pressed',CONTROL),('active',CONTROL)],darkcolor=[('pressed',CONTROL),('active',CONTROL)])
    s.configure('On.TButton',background=CONTROL,foreground=ACCENT,bordercolor=ACCENT,lightcolor=CONTROL,darkcolor=CONTROL,padding=(10,7))
    s.map('On.TButton',background=[('pressed',HOVER),('active',HOVER)])

    s.configure('TPanedwindow',background=PANEL)
    s.configure('Sash',background=BORDER,sashthickness=6,gripcount=0,lightcolor=PANEL,darkcolor=PANEL,bordercolor=PANEL)

    return fonts

def slider(parent,variable,low,high,step,length=None):
    """ttk's themed Scale keeps a pale trough outline here, so draw a plain Tk one instead."""
    scale=tk.Scale(parent,variable=variable,from_=low,to=high,resolution=step,orient='horizontal',
                   showvalue=0,background=HANDLE,activebackground=TEXT,troughcolor=WELL,
                   highlightthickness=0,borderwidth=0,sliderrelief='flat',sliderlength=22,width=10)
    if length is not None:scale.configure(length=length)
    return scale

def style_toolbar(toolbar,fonts):
    """Repaint matplotlib's Tk toolbar to sit on the dark ground.

    macOS draws tk.Button with the native face and ignores -background, while cget still
    reports whatever we set. Matplotlib picks its icon colour from that reported value, so
    darkening the buttons there would leave pale glyphs on a pale face; leave them native.
    """
    native_button_face=toolbar.tk.call('tk','windowingsystem')=='aqua'
    toolbar.configure(background=PANEL,borderwidth=0,highlightthickness=0)
    for child in toolbar.winfo_children():
        kind=child.winfo_class()
        if kind in ('Button','Checkbutton'):
            child.configure(highlightbackground=PANEL,borderwidth=0)
            if native_button_face:continue
            child.configure(background=PANEL,foreground=TEXT,activebackground=HOVER,activeforeground=TEXT)
            if kind=='Checkbutton':child.configure(selectcolor=CONTROL)
            if getattr(child,'_image_file',None):toolbar._set_image_for_button(child)
        elif kind=='Label':child.configure(background=PANEL,foreground=MUTED,font=fonts['small'])
        elif kind=='Frame':child.configure(background=BORDER)

def style_listbox(listbox,fonts):
    listbox.configure(background=WELL,foreground=TEXT,selectbackground=SELECT,selectforeground=TEXT,font=fonts['ui'],borderwidth=0,highlightthickness=1,highlightbackground=BORDER,highlightcolor=BORDER,activestyle='none')
