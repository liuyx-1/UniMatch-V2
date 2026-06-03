"""Generate a Chinese-language speech script PDF for the
'术中视频场景分割' presentation.

Order: 背景 → 实验结果 → 方法 → 未来研究计划.

Usage:
    pip install reportlab
    python tools/make_speech_pdf.py
    # → outputs ./speech_surgical_seg.pdf in the current directory
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Register a built-in CJK font (no external file required)
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
CJK = 'STSong-Light'

OUT = 'speech_surgical_seg.pdf'

# ---------- styles ----------
ss = getSampleStyleSheet()

style_title = ParagraphStyle(
    'TitleCN', parent=ss['Title'],
    fontName=CJK, fontSize=20, leading=26,
    textColor=HexColor('#1a3a5c'),
    spaceAfter=14, alignment=1,
)
style_section = ParagraphStyle(
    'SectionCN', parent=ss['Heading2'],
    fontName=CJK, fontSize=14, leading=20,
    textColor=HexColor('#2c5f8c'),
    spaceBefore=14, spaceAfter=8,
)
style_body = ParagraphStyle(
    'BodyCN', parent=ss['BodyText'],
    fontName=CJK, fontSize=11.5, leading=20,
    firstLineIndent=24,
    spaceAfter=8,
    alignment=4,         # 4 = TA_JUSTIFY
)
style_body_noindent = ParagraphStyle(
    'BodyCN_NoIndent', parent=style_body,
    firstLineIndent=0,
)
style_meta = ParagraphStyle(
    'MetaCN', parent=ss['BodyText'],
    fontName=CJK, fontSize=9.5, leading=14,
    textColor=HexColor('#666666'),
    spaceAfter=12, alignment=1,
)

# ---------- speech content ----------
TITLE = '术中视频场景分割 · 演讲文案'
META  = '汇报顺序：研究背景 → 实验结果 → 研究方法 → 未来研究计划　·　建议时长 ≈ 4 分钟'

SECTIONS = [
    # ---------- 1. 背景 ----------
    ('一、研究背景与意义',
     [
        '各位老师好，我汇报的研究方向是<b>术中视频场景分割</b>，'
        '这是支撑手术机器人感知、自主操作与术中导航的关键视觉任务。',

        '腹腔镜手术的视觉环境极为复杂——组织、器械、烟雾、血液、遮挡常常交织在一起，'
        '给场景分割带来三重挑战：'
        '<b>第一，场景结构复杂</b>，多种解剖与器械相互遮挡；'
        '<b>第二，类别细粒度且严重长尾</b>，胆囊三角、胆囊管等关键解剖结构边界模糊、'
        '像素占比往往不到 1%；'
        '<b>第三，像素级标注成本极高</b>，逐帧精细标注一段术中视频可能需要数小时。',

        '因此本研究的核心目标是：<b>在仅有少量像素级标注的条件下，'
        '构建稳定、可泛化的术中视频场景分割模型</b>，'
        '为后续的关键点跟踪、术中导航与机器人自主操作提供可靠的视觉基础。'
        '这条路线一旦走通，将在结构感知稳定性、关键点跟踪能力、以及标注依赖降低三个层面同时获益。',
     ]),

    # ---------- 2. 实验结果（含痛点+策略简述） ----------
    ('二、实验结果',
     [
        '我们在 <b>4 个手术分割主流数据集</b> 上开展了系统实验，'
        '涵盖胆囊切除（Endoscapes-Seg50、CholecSeg8k）与机器人手术（EndoVis2018、EndoVis2017），'
        '类别数从 4 到 13、图像总量近 1.5 万帧、跨 96 段手术视频。',

        '<b>实验中暴露出三个关键痛点。</b>'
        '第一，强基线 UniMatch-V2 在 Endoscapes-Seg50 上仅得到 46.56 mIoU，'
        '稀有解剖类（如 cystic_plate、calot_triangle）几乎无法识别——'
        '<b>长尾问题严重</b>；'
        '第二，全量微调 DINOv2-B 等大模型会快速过拟合且显存占用大——'
        '<b>缺乏高效的领域适配机制</b>；'
        '第三，细管腔与器械尖端的边界经常被模糊或漏分——'
        '<b>结构边缘信号利用不足</b>。',

        '<b>针对这三大痛点，我们提出了 4 个互补模块的统一框架。</b>'
        '具体来说：用 <b>HPTA</b> 冻结大模型主干、仅训练轻量适配器来解决领域适配与显存压力；'
        '用 <b>LC-PAM</b> 把医学语言先验直接注入像素级亲和，为稀有解剖类补充类别语义；'
        '用 <b>EABR</b> 同时在特征侧与输出侧利用边缘信号，强化细结构边界；'
        '用 <b>EPLA</b> 在线维护类先验、对伪标签做去偏，防止尾类自饿。'
        '——这些方法的具体设计将在下一部分详细介绍。',

        '<b>消融结果显示</b>：base + HPTA 把 mIoU 从 46.56 提升到 47.32；'
        '加入 LC-PAM 文本分支后<b>单项达到 49.16 mIoU、mDice 58.43</b>，是所有变体中最优——'
        '这正是语言先验在尾类上的直接体现，'
        'cystic_plate 与 calot_triangle 的 IoU 实现明显跃升；'
        'EABR boundary 分支单独贡献了最高的 PA（95.28%）与 fwIoU（91.60%）。'
        '在 EndoVis2018 与 Endoscapes-seg50 的视频可视化上，'
        '模型对器械尖端与胆囊轮廓保持时间一致的稳定跟踪，'
        '证明该框架在连续帧上具有实用价值。',
     ]),

    # ---------- 3. 方法（详细） ----------
    ('三、研究方法（详解）',
     [
        '我们以 <b>UniMatch-V2</b> 为半监督分割基线，在其之上提出了 4 个模块化创新，'
        '构成一条单一推理链路：'
        '<b>HPTA → LC-PAM → EABR → 分割头</b>，'
        '并将 <b>EPLA</b> 作为伪标签路径上的训练时旁路修正。',

        '<b>① HPTA（Hierarchical Patch-Token Adapter）</b>：'
        '完全冻结 DINOv2-B 主干，在四个 DPT 对齐层 {l<sub>1</sub>..l<sub>4</sub>} 并行插入残差适配器，'
        '内部采用 local 深度卷积 + global MLP 双分支加 softmax 门控融合，γ 零初始化保证训练稳定，'
        '每层仅引入 O(d<sup>2</sup>/r)（r=8）可训参数，既保留基础模型先验又显著降低显存开销，'
        '是后续所有模块的稳定特征底座。',

        '<b>② LC-PAM（Language-Anchored Class-Pixel Affinity）</b>：'
        '用冻结的 SigLIP2 编码每类 4 条手写医学 prompt，'
        '经投影头与 L2 归一化得到非背景类的语言锚点 T ∈ ℝ<sup>(C−1)×128</sup>；'
        '可训练视觉投影 φ 把 patch token 映射到同空间，'
        '构造类–像素余弦亲和度 A<sub>c,i</sub>=⟨T<sub>c</sub>, φ(z<sub>i</sub>)⟩/τ；'
        '再经 Top-k 池化得到类先验 P，并以 per-pixel sigmoid 门控做凸组合 '
        'z′=(1−g)⊙z + g⊙P 融入 DPT logits，背景通道由非背景掩码屏蔽。'
        '该模块为稀有解剖结构注入语言语义，是<b>本研究最核心的创新点</b>。',

        '<b>③ EABR（Edge-Aware Boundary Refinement）</b>：'
        '从 RGB 提取 Canny 边缘 e，分两路利用——'
        '<b>特征侧 DREC</b> 通过两个零初始化适配器 A<sub>t</sub>、A<sub>v</sub> '
        '同时残差注入文本 latent p<sup>lat</sup> 与 DPT 前的视觉特征 F<sub>vis</sub>；'
        '<b>输出侧 MA-CBA</b> 用一个 mini-head g<sub>ψ</sub>（3×3 → 1×1 Conv）'
        '把融合 logits 映射为单通道边界图 b，由 Sobel + 形态学膨胀派生的 b* 监督，'
        '损失为加权 BCE+Dice，CutMix 边带 δ=2 px 屏蔽避免拼贴边被误学。'
        '两路在同一边缘信号上形成 <b>input → feature → output 闭环</b>，'
        '专门强化胆管、血管、器械尖端等手术细结构。',

        '<b>④ EPLA（EMA-Prior Logit Adjustment）</b>：'
        '由标注像素初始化、并按 EMA 在线更新类先验 π̂（ρ=0.99）；'
        '训练时对 logits 做 Menon 风格调整 z̃ = z − α<sub>t</sub> log π̂ ；'
        '关键技巧是<b>先验由未调整的 z 更新、伪标签则用 z̃ 计算</b>，'
        '防止尾类自饿，显著缓解长尾伪标签遗漏现象。'
        'EPLA 仅作用于训练时的伪标签生成路径，不改变推理 forward。',
     ]),

    # ---------- 4. 未来 ----------
    ('四、未来研究计划',
     [
        '我们将在两个方向继续推进：',
        '<b>① 增强去偏机制</b>：在 EPLA 基础上探索动态阈值与类别自适应温度，'
        '并融合对比学习，进一步压缩头-尾类的 gap<sub>HT</sub>，'
        '提升 cystic_plate / calot_triangle 等稀有解剖结构的分割稳定性。',
        '<b>② 跨数据集泛化</b>：将完整框架迁移至 CholecSeg8k、EndoVis2018 等数据集，'
        '系统验证 LC-PAM 文本先验与 EABR 边缘耦合的跨域迁移能力，'
        '并构建多数据集联合训练的统一手术场景分割基准。',
        '汇报完毕，恳请各位老师批评指正，谢谢！',
     ]),
]

# ---------- build PDF ----------
doc = SimpleDocTemplate(
    OUT, pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2.0*cm, bottomMargin=2.0*cm,
    title=TITLE, author='术中视频场景分割',
)

flow = []
flow.append(Paragraph(TITLE, style_title))
flow.append(Paragraph(META,  style_meta))
flow.append(Spacer(1, 0.2*cm))

for header, paras in SECTIONS:
    flow.append(Paragraph(header, style_section))
    for p in paras:
        flow.append(Paragraph(p, style_body))
    flow.append(Spacer(1, 0.15*cm))

doc.build(flow)
print(f'saved {OUT}')
