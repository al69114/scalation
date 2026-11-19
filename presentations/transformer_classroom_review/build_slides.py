"""Build an editable six-slide PowerPoint and matching vector PDF.

Run with python-pptx, reportlab, and pymupdf installed.
All diagrams are native shapes. The Scala implementation is read-only source material.
"""
from pathlib import Path
import math
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

OUT = Path(__file__).resolve().parent
W, H = 960, 540
NAVY, TEAL, ORANGE = '142D40', '087F83', 'D96535'
INK, MUTED, LINE = '213D4C', '5B707C', 'DCE5E9'
BG, PALE, MINT, CREAM = 'F6F8FA', 'EAF0F5', 'E7F5F2', 'FFF1E8'
WHITE = 'FFFFFF'
fonts = Path('/System/Library/Fonts/Supplemental')
pdfmetrics.registerFont(TTFont('Arial', str(fonts / 'Arial.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold', str(fonts / 'Arial Bold.ttf')))
pdfmetrics.registerFont(TTFont('Menlo', '/System/Library/Fonts/Menlo.ttc', subfontIndex=0))
prs = Presentation()
prs.slide_width, prs.slide_height = Pt(W), Pt(H)
prs.core_properties.title = 'Transformers: From Attention to ScalaTion'
prs.core_properties.subject = 'Six-slide classroom review of transformer concepts and SimpleTransformer.scala'
prs.core_properties.author = 'Classroom Review'
pdf = canvas.Canvas(str(OUT / 'transformer_review.pdf'), pagesize=(W, H))
pdf.setTitle(prs.core_properties.title)
pdf.setAuthor('Classroom Review')
slide = None
notes = []
checks = []

def rect(x,y,w,h,fill=WHITE,line=None,radius=False):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Pt(x),Pt(y),Pt(w),Pt(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line: sh.line.color.rgb = RGBColor.from_string(line)
    else: sh.line.fill.background()
    if radius:
        sh.adjustments[0] = .12
    pdf.setFillColor(HexColor('#'+fill))
    pdf.setStrokeColor(HexColor('#'+(line or fill)))
    if radius: pdf.roundRect(x,H-y-h,w,h,8,stroke=bool(line),fill=1)
    else: pdf.rect(x,H-y-h,w,h,stroke=bool(line),fill=1)
    return sh

def text(x,y,w,content,size=18,color=INK,bold=False,font='Arial',leading=None):
    leading = leading or size*1.25
    lines = content.split('\n')
    height = len(lines)*leading + 4
    sh = slide.shapes.add_textbox(Pt(x),Pt(y),Pt(w),Pt(height))
    tf = sh.text_frame
    tf.clear(); tf.word_wrap=False
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
    tf.vertical_anchor=MSO_ANCHOR.TOP
    pf = 'Menlo' if font=='Menlo' else ('Arial-Bold' if bold else 'Arial')
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=line; p.font.name=font; p.font.size=Pt(size); p.font.bold=bold
        p.font.color.rgb=RGBColor.from_string(color)
        p.space_before=p.space_after=Pt(0); p.line_spacing=Pt(leading)
        pdf.setFont(pf,size); pdf.setFillColor(HexColor('#'+color))
        pdf.drawString(x,H-y-size-i*leading,line)
        width = pdfmetrics.stringWidth(line,pf,size)
        checks.append((len(prs.slides),line,width,w))
        assert width <= w+1, (len(prs.slides),line,width,w)
    assert y+height <= H, (content,y,height)
    return sh

def arrow(x,y,w,color=TEAL):
    sh=slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,Pt(x),Pt(y),Pt(w),Pt(13))
    sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor.from_string(color); sh.line.fill.background()
    pdf.setFillColor(HexColor('#'+color))
    p=pdf.beginPath(); p.moveTo(x,H-y-4);p.lineTo(x+w-7,H-y-4);p.lineTo(x+w-7,H-y)
    p.lineTo(x+w,H-y-6.5);p.lineTo(x+w-7,H-y-13);p.lineTo(x+w-7,H-y-9)
    p.lineTo(x,H-y-9);p.close();pdf.drawPath(p,fill=1,stroke=0)

def begin(n,section,title,subtitle,source):
    global slide
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    rect(0,0,W,H,BG);rect(0,0,W,7,TEAL)
    text(40,24,850,'TRANSFORMERS  /  CLASSROOM REVIEW  /  '+section,10,TEAL,True)
    text(40,52,885,title,30,NAVY,True)
    text(40,97,885,subtitle,16,MUTED)
    rect(40,494,880,1,LINE)
    text(40,507,827,source,9,MUTED)
    text(883,505,42,f'{n} / 6',11,TEAL,True)

def end(note):
    slide.notes_slide.notes_text_frame.text=note
    notes.append(note)
    pdf.showPage()

def label(x,y,w,heading,body,fill=MINT,accent=TEAL):
    rect(x,y,w,100,fill,radius=True)
    text(x+16,y+12,w-32,heading,20,accent,True)
    text(x+16,y+44,w-32,body,16,INK)

begin(1,'THE BIG IDEA','Transformers connect parts of a sequence',
      'Self-attention lets each input gather useful information from other inputs.',
      'Source: Miller, Introduction to Computational Data Science Using ScalaTion, §11.10, pp. 560–564.')
rect(40,145,547,295,WHITE,radius=True)
text(60,164,500,'A forecasting view',20,NAVY,True)
text(60,199,500,'Each token can be a short patch of a time series.',16,MUTED)
for i in range(4):
    x=62+i*126
    rect(x,245,108,52,MINT,radius=True)
    text(x+16,260,82,f'Patch {i+1}',17,TEAL,True)
rect(62,326,486,53,NAVY,radius=True)
text(84,340,440,'Self-attention mixes information across patches',17,WHITE,True)
text(62,399,480,'Past patches  →  contextual features  →  forecast',16,TEAL,True)
label(612,145,308,'Connect','Compare distant parts\nof the input directly.')
label(612,258,308,'Keep order','Position encodings identify\nwhere each patch belongs.',PALE,NAVY)
rect(612,371,308,69,CREAM,radius=True)
text(628,385,278,'Our code: one encoder block\nwith one attention head.',17,ORANGE,True)
text(40,459,880,'Review path: attention → encoder block → Scala shapes → training',17,NAVY,True)
end('''Purpose (about 1 minute): Introduce self-attention as a learned way to combine sequence elements. The book contrasts this with recurrent models, which propagate a hidden state through time. A transformer encoder can compute attention across its observed input window without recurrence.

In this implementation, a token is a non-overlapping patch of scalar observations, not a word. Every patch can attend to every observed patch. That is compatible with forecasting because all input patches precede the forecast origin. The target future patch must stay outside the input.

Scope: This is an encoder-only forecaster, with a linear prediction head and no decoder. It is a single-head teaching implementation; a full transformer can use multiple heads and stacked layers. Do not assume all transformers have the same architecture.

Source: John A. Miller, Introduction to Computational Data Science Using ScalaTion, April 9, 2026, §11.10, actual PDF/printed pp. 560–565. The table of contents lists earlier page numbers (536 onward); the deck cites the page numbers on the actual pages. Local code: src/main/scala/scalation/modeling/SimpleTransformer.scala, class declaration at line 298 and encoderForward at line 340. Code reviewed as supplied; no model run or benchmark is claimed.''')

begin(2,'SELF-ATTENTION','Q, K, and V turn similarity into context',
      'The same input X is projected into three learned views: Q = XWQ, K = XWK, V = XWV.',
      'Sources: textbook §11.10.1, pp. 560–562, Eq. 11.59; SimpleTransformer.scala:174–181.')
for x,h,b in [(40,'Query Q','What am I looking for?'),(340,'Key K','What can I match with?'),(640,'Value V','What information do I share?')]:
    rect(x,144,280,74,WHITE,radius=True)
    text(x+15,155,250,h,20,TEAL,True);text(x+15,186,252,b,15,INK)
rect(40,237,880,70,NAVY,radius=True)
text(64,253,832,'A = softmax(QKᵀ / √(d_k))          Context C = AV',27,WHITE,True)
text(41,327,425,'Compare → scale → normalize → combine',17,TEAL,True)
text(41,360,425,'Each row of A sums to 1.\n√(d_k) reduces large dot-product magnitudes.',16,INK,leading=25)
rect(40,423,425,48,MINT,radius=True)
text(53,436,398,'Example: 0.7 × 10 + 0.3 × 20 = 13',18,TEAL,True)
rect(489,327,431,144,NAVY,radius=True)
text(505,338,399,'attention(...) — source excerpt',12,'9DDAD5',True)
text(505,366,399,'val qkt = q * k.ᵀ\nval sdp = qkt / sqrt (d_k)\nval scr = f_softmax.fM (sdp)\n(scr, scr * v)',14,WHITE,font='Menlo',leading=22)
end('''Purpose (about 2 minutes): Explain that queries and keys determine relevance, while values carry the information that will be combined. All three come from the same input in self-attention; their projection matrices differ and are learned.

For L patches and model width d, this code has Q, K, V of shape L × d. QKᵀ and the attention probability matrix A have shape L × L. Multiplying A by V returns L × d. Softmax is applied separately to each row; the local AFF.fM implementation uses row-wise matricize. The scaling factor is the square root of the query/key width d_k, not the sequence length. Here d_k = d_mod because there is one head.

The scalar example uses assumed attention weights 0.7 and 0.3 and assumed values 10 and 20 to illustrate a weighted sum. These are not measured model outputs. With vector-valued V, exactly the same weighted combination is applied to every value feature. Ask students which value contributes more and why the result is between 10 and 20.

Distinguish attention weights A (called scr in Scala) from attention output C (scr * v). The method returns both so the backward pass can reuse the probabilities. The displayed code omits the debug print but otherwise preserves the relevant statements.

Sources: textbook §11.10.1, pp. 560–562, especially Eq. 11.59 on p. 561; SimpleTransformer.scala lines 174–181; ActivationFun.scala AFF definition; MatrixD.scala matricize definition.''')

begin(3,'THE ENCODER','Position, attention, and a feed-forward network',
      'The implementation uses two residual additions, each followed by layer normalization.',
      'Sources: textbook §§11.10.2–11.10.3, pp. 562–564, Fig. 11.9; SimpleTransformer.scala:127, 225, 261, 340.')
steps=[('X + PE','Input order',MINT,TEAL),('Attention','Mix patches',PALE,NAVY),('Add + Norm','Keep + normalize',MINT,TEAL),('FFN','Transform features',PALE,NAVY),('Add + Norm','Encoder output',MINT,TEAL)]
for i,(h,b,f,c) in enumerate(steps):
    x=40+i*180
    rect(x,153,160,83,f,radius=True)
    text(x+12,166,138,h,18,c,True);text(x+12,201,138,b,12,MUTED)
    if i<4: arrow(x+163,187,14)
rect(40,260,430,201,WHITE,radius=True)
text(58,277,395,'Position gives a patch its location',19,TEAL,True)
text(58,316,395,'Sin/cos patterns are added to the input.\nWithout position signals, self-attention\nalone cannot identify sequence order.',17,INK,leading=26)
text(58,408,395,'encodePositions(lookback, d_mod)',15,TEAL,font='Menlo')
rect(490,260,430,201,WHITE,radius=True)
text(508,277,395,'The block, in equations',19,NAVY,True)
text(508,313,395,'Z = LN(U + Attention(U))\nO = LN(Z + FFN(Z))\nFFN(Z) = ReLU(ZW1)W2',20,INK,leading=33)
text(508,421,395,'U = X + PE; LN normalizes each row.',15,MUTED)
end('''Purpose (about 2 minutes): Walk left to right. Position encoding is added first. The first sublayer combines information across patches using attention. The second sublayer applies the same feed-forward network to each patch independently. The residual paths add the sublayer input back before normalization; the equations show exactly what is added. This is post-layer normalization.

encodePositions uses PE(p,2i) = sin(p / 10000^(2i/d_mod)) and PE(p,2i+1) = cos(p / 10000^(2i/d_mod)). The demo has d_mod = 10, so all columns form sin/cos pairs. The current loop leaves an unfilled final column for odd widths. The standalone embedding routines are commented out: raw patch length is the model width in the demo.

The FFN expands each row from 10 to 40 features, applies ReLU, and returns it to 10 features. There are no bias terms or dropout in this implementation. The normalization scale and shift default to fixed scalar values 1 and 0; they are not learned feature vectors here. Multi-head attention would compute several projected attention outputs and combine them, but this code uses only one head.

Implementation detail for technical discussion: layerNorm calls MatrixD.variance, which delegates to VectorD.variance and uses a sample-variance denominator N-1. layerNormBackward uses the usual population-variance derivative form with N. These should be made consistent and checked with finite differences before relying on training. This is a source-review finding, not a training result.

Sources: textbook §§11.10.2–11.10.3, pp. 562–564, including Fig. 11.9; SimpleTransformer.scala encodePositions at line 127, layerNorm at 225, layerNormBackward at 239, ffn at 261, encoderForward at 340; MatrixD.scala variance at 1696 and VectorD.scala variance at 1375.''')

begin(4,'FROM SERIES TO MATRICES','The demo uses 40 past values to predict 10',
      'Example_LakeLevels → non-overlapping patches → sliding windows → future target vector.',
      'Source: SimpleTransformer.scala:43–49, 88–120, 476–499 (simpleTransformerTest).')
text(40,144,880,'ONE TRAINING EXAMPLE',11,TEAL,True)
for i in range(5):
    x=40+i*180;future=i==4
    rect(x,169,160,81,CREAM if future else MINT,radius=True)
    text(x+15,181,134,'Target' if future else f'Patch {i+1}',20,ORANGE if future else TEAL,True)
    text(x+15,216,134,f'Values {i*10+1}–{(i+1)*10}',15,INK)
    if i==3: arrow(x+164,201,13,ORANGE)
text(40,269,700,'X: 4 patches × 10 values = 40 historical observations',18,TEAL,True)
text(760,269,160,'y: 10 future values',15,ORANGE,True)
rect(40,312,880,150,WHITE,radius=True)
rows=[('Stage','Shape','Code connection'),('Input / Q, K, V','4 × 10','lookback = 4; d_mod = 10'),('Attention weights','4 × 4','one weight per query–key patch pair'),('FFN / encoder output','4 × 40 → 4 × 10','d_ff = 40'),('Flatten / prediction','40 → 10','wP is 40 × 10; horizon = 10')]
for i,row in enumerate(rows):
    y=322+i*27
    if i==0: rect(49,y-2,862,25,PALE)
    for x,w,s in zip([58,351,554],[275,185,350],row):
        text(x,y,w,s,14 if i else 13,NAVY if i==0 else INK,i==0)
end('''Purpose (about 1.5 minutes): Tie every dimension to the supplied runnable example. patchLength = 10, lookPatches = 4, forePatches = 1. Thus d_mod = 10, d_ff = 40, and horizon = 10. A forecast predicts ten scalar time steps directly, not one scalar or one token ID.

The first illustrated window consumes observations 1–40 and targets 41–50; the next window shifts one patch, consuming 11–50 and targeting 51–60. These are illustrative one-based indices, not real lake-level values. Neighboring examples overlap even though the patches themselves do not.

patchify creates floor(m / patchLength) full patches and drops any incomplete trailing patch. createSlidingWindowDataset produces P - lookback - forePatches + 1 examples when there are enough patches. Targets are flattened into a VectorD of length forePatches * patchLength.

The attention matrix is 4 × 4 because tokens are patches, not individual scalar observations. The FFN expands feature width without changing patch count. The final encoder output is flattened to length 40, then multiplied by wP of shape 40 × 10 to produce a horizon-length vector.

Source: SimpleTransformer.scala patchify lines 43–49, createSlidingWindowDataset lines 88–120, demo configuration and data preparation lines 476–499. Units of lake observations are deliberately not asserted here; the slide uses time steps.''')

begin(5,'THE SCALA WALKTHROUGH','Forward pass → loss → backward pass → update',
      'The prediction head turns encoder features into future values; training learns all six weight matrices.',
      'Source: SimpleTransformer.scala:340–357, 364–405, 416–454, 501–504. Code excerpts omit comments.')
rect(40,144,557,327,NAVY,radius=True)
text(58,158,520,'encoderForward(...) — key statements',12,'9DDAD5',True)
code='''val (q, k, v) = (x * wQ, x * wK, x * wV)
val (scr, attn_out) = attention (q, k, v, d_mod)
val x_norm1 = layerNorm (x + attn_out)

val (ffn_out, h_linear) = ffn (x_norm1, w1, w2)
val out = layerNorm (x_norm1 + ffn_out)

val flattened = out.flatten
val pred = flattened *: wP'''
text(58,197,520,code,13,WHITE,font='Menlo',leading=26)
text(58,446,520,'Excerpt begins after positional encoding is added.',12,'9DDAD5')
rect(620,144,300,327,WHITE,radius=True)
text(638,160,265,'Learning from error',21,TEAL,True)
text(638,205,265,'1   Compare forecast with target\n2   Backpropagate gradients\n3   Update weights immediately',15,INK,leading=30)
rect(635,309,270,64,MINT,radius=True)
text(650,322,242,'MSE = mean((prediction − y)²)\nW ← W − learningRate × gradient',13,TEAL,True,leading=23)
text(638,395,260,'Demo settings',15,NAVY,True)
text(638,422,260,'40 epochs; learning rate 0.007',15,INK)
end('''Purpose (about 2 minutes): Map the conceptual block to the actual forward-pass statements. The line x += posEnc executes immediately before this excerpt and is discussed on slide 6. In ScalaTion, flattened *: wP denotes the vector–matrix multiplication that produces the forecast. The method returns EncoderCache, retaining intermediate tensors for manual backpropagation.

The training loop processes one training example at a time in sequence. It computes error = pred - target and MSE = (error dot error) / target.dim. The derivative seed is d_pred = error * (2.0 / target.dim). encoderBackward propagates this through the prediction head, both normalization layers, FFN, attention, and Q/K/V projections. Residual connections require adding gradients from both paths.

The six trainable matrices updated after every instance are wQ, wK, wV, w1, w2, and wP. This is incremental gradient descent, not a mini-batch optimizer or Adam. The supplied demo sets learningRate = 0.007 and totalEpochs = 40. These are example settings, not tuned recommendations or evidence of convergence.

Code-review status: The supplied source was inspected, but the training program was not executed for this presentation. No training loss, test accuracy, or forecast plot is invented. The final sample trace in the source is from training data, as explained on the next slide.

Sources: SimpleTransformer.scala encoderForward lines 340–357, encoderBackward lines 364–405, train lines 416–454, demo settings lines 501–504.''')

begin(6,'CLASSROOM CHECK','Explain the mechanism—and read the demo critically',
      'Use these questions to check understanding before discussing forecasting performance.',
      'Sources: textbook §11.10, pp. 560–565; SimpleTransformer.scala:225–253, 340–341, 506–521; VectorD.scala:1375.')
rect(40,144,421,327,WHITE,radius=True)
text(58,162,385,'Three review questions',21,TEAL,True)
questions=[('01','Why divide QKᵀ by √(d_k)?','Think about score size and softmax.'),('02','Why add positional encoding?','Think about patch order.'),('03','Why is attention 4 × 4 here?','Think about what counts as a token.')]
for i,(n,q,hint) in enumerate(questions):
    y=211+i*80
    text(58,y,35,n,18,TEAL,True)
    text(101,y,345,q,17,NAVY,True)
    text(101,y+29,345,hint,14,MUTED)
rect(485,144,435,327,CREAM,radius=True)
text(503,162,399,'Three implementation caveats',21,ORANGE,True)
for i,(h,b) in enumerate([
    ('Repeated calls change the input','x += posEnc accumulates position signals.'),
    ('The final sample is from training','xTrain.head does not measure generalization.'),
    ('Normalization needs a gradient check','Forward/backward variance conventions differ.')]):
    y=212+i*80
    text(503,y,399,h,17,NAVY,True)
    text(503,y+29,399,b,14,INK)
end('''Purpose (about 2 minutes): Ask the questions before revealing the answers verbally.

Answers:
1. Scaling by sqrt(d_k) counteracts the tendency of dot-product magnitudes to grow with feature width, helping avoid excessively sharp softmax probabilities and tiny gradients. It does not itself normalize the attention rows; softmax does that.
2. Position encodings supply location information. Content-only self-attention is permutation-equivariant; it cannot itself know which position a patch occupies. The subsequent fixed flattening head can depend on position, but that is distinct from giving attention position information.
3. Four input patches create four queries and four keys. Thus QKᵀ is 4 × 4 even though each patch has ten features and the forecast contains ten scalar observations.

Code caveats, from static inspection:
• encoderForward uses x += posEnc (line 341). MatrixD.+= mutates the matrix. train repeatedly passes the same matrices from trainingData, so position encodings accumulate over epochs; the later sample trace adds them again. A repair would use a fresh local sum and consistently use it in forward calculations and the cache. No source changes were made for this deck.
• The sampleInput = xTrain.head and yy = yTrain.head evaluation uses a training example. It is not held-out validation, despite its printed label. For a performance study, use chronological training/validation/test splits, ensure training targets precede the split, fit any preprocessing on training data, and compare with appropriate simple forecasting baselines.
• layerNorm computes sample variance through MatrixD.variance → VectorD.variance (denominator N-1), while layerNormBackward uses the usual N-based normalization derivative. Align the forward and backward convention and verify gradients numerically before relying on optimization results. No numerical gradient check was run for this slide deck.

Reference list:
John A. Miller. Introduction to Computational Data Science Using ScalaTion. April 9, 2026. Supplied file comp_data_science (1).pdf, §11.10 and Fig. 11.9, actual pp. 560–565; related attention discussion §11.9.3 on pp. 558–559.
Local SimpleTransformer.scala, John A. Miller, header dated November 12, 2026; src/main/scala/scalation/modeling/SimpleTransformer.scala. Supporting implementation checks: MatrixD.scala and VectorD.scala in scalation/mathstat, plus ActivationFun.scala. This later local implementation supplements the book; it is not claimed to be printed in the April PDF.

Suggested closing question: Why can an unmasked encoder still be used for forecasting here? Answer: it attends only over the observed historical window; future targets are not part of the input.''')

pdf.save()
prs.save(OUT/'transformer_review.pptx')
md=['# Transformer classroom review — speaker notes','',
    'Six slides, approximately 10–12 minutes. The deck is based on the supplied textbook and local Scala implementation. Code was reviewed, not executed.','']
for i,note in enumerate(notes,1): md.extend([f'## Slide {i}', '', note, ''])
(OUT/'speaker_notes.md').write_text('\n'.join(md))

# Check saved structure and render the companion PDF for a visual review.
import pymupdf as fitz
from PIL import Image, ImageOps, ImageDraw
assert len(Presentation(OUT/'transformer_review.pptx').slides)==6
doc=fitz.open(OUT/'transformer_review.pdf')
assert len(doc)==6
thumbs=[]
for i,p in enumerate(doc):
    pix=p.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
    path=OUT/f'preview_{i+1}.png';pix.save(path)
    im=Image.open(path).resize((720,405))
    thumbs.append(im)
contact=Image.new('RGB',(1470,1260),'#DCE5E9')
for i,im in enumerate(thumbs): contact.paste(im,(10+(i%2)*735,10+(i//2)*415))
contact.save(OUT/'slide_overview.png')
print(f'Created 6 slides; verified {len(checks)} text lines fit their boxes.')
print(OUT/'transformer_review.pptx')
print(OUT/'transformer_review.pdf')
