"""Relabel Figs. 4 and 5 (text-only edits on white background; image content untouched).
RealESRGAN -> Real-ESRGAN, Notumor -> No tumor, 'Mean Error=' -> 'MAE:', slightly larger fonts.
Usage: python scripts/relabel_figs45.py figures/notebook_raw figures
(expects the raw notebook files renamed to fig4_glioma_detail.png = fig_qualitative_mri_0010.png and
 fig5_residuals_allclasses.png = fig_qualitative_4classes.png)"""
import sys
from PIL import Image, ImageDraw, ImageFont
src,out=sys.argv[1],sys.argv[2]
D='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
S=52
H=lambda k: ImageFont.truetype(D,int(S*k))
im=Image.open(f'{src}/fig4_glioma_detail.png').convert('RGB'); d=ImageDraw.Draw(im)
hdr={'GT (HR)':(293,495),'Bicubic':(1021,1206),'EDSR':(1762,1899),'Waifu2x':(2442,2653),'Real-ESRGAN':(3099,3429),'SwinIR':(3897,4068)}
d.rectangle((0,0,im.width,88),fill='white')
for t,(a,b) in hdr.items(): d.text(((a+b)/2,70),t,font=H(1.3),fill='black',anchor='ms')
d.rectangle((700,2150,4000,2235),fill='white')
for v,(a,b) in [('4.02',(860,1243)),('3.11',(1534,1917)),('3.20',(2209,2593)),('8.60',(2883,3267)),('3.24',(3557,3942))]:
    d.text(((a+b)/2,2198),f'MAE: {v}',font=H(1.0),fill='black',anchor='ms')
im.save(f'{out}/fig4_glioma_detail.png',dpi=(300,300))

im=Image.open(f'{src}/fig5_residuals_allclasses.png').convert('RGB'); d=ImageDraw.Draw(im)
hdr={'GT (HR)':(386,588),'Bicubic':(1276,1461),'EDSR':(2181,2320),'Waifu2x':(3023,3235),'Real-ESRGAN':(3844,4174),'SwinIR':(4804,4978)}
d.rectangle((0,0,im.width,88),fill='white')
for t,(a,b) in hdr.items(): d.text(((a+b)/2,76),t,font=H(1.35),fill='black',anchor='ms')
# enlarge MAE labels: erase old (below each panel row) and redraw
import numpy as np
rows=[(867,917),(1739,1789),(2611,2661),(3483,3534)]
cols=[(982,1756),(1862,2636),(2742,3516),(3623,4397),(4503,5277)]
vals=[['4.02','3.11','3.20','8.60','3.24'],['3.96','3.18','3.18','6.69','3.17'],['6.43','4.84','5.05','10.47','4.74'],['4.55','3.52','3.58','7.16','3.54']]
for r,(y0,y1) in enumerate(rows):
    d.rectangle((900,y0-4,5300,min(y1+14,im.height)),fill='white')
    for c,(x0,x1) in enumerate(cols):
        d.text(((x0+x1)/2,y0+48),f'MAE: {vals[r][c]}',font=H(1.1),fill='black',anchor='ms')
# row label No tumor
d.rectangle((20,2060,95,2400),fill='white')
f0=H(1.0); txt='No tumor'; tw=int(f0.getlength(txt))+10
tmp=Image.new('RGB',(tw,80),'white'); ImageDraw.Draw(tmp).text((tw/2,60),txt,font=f0,fill='black',anchor='ms')
rot=tmp.rotate(90,expand=True); cy=(2104+2353)/2
im.paste(rot,(int(58-rot.width/2),int(cy-rot.height/2)))
im.save(f'{out}/fig5_residuals_allclasses.png',dpi=(300,300))
