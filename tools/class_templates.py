"""Hand-written medical-domain text prompt templates per global class.

Usage in code:
    from tools.class_templates import TEMPLATES
    prompts_for_class_c = TEMPLATES[global_name_of_class_c]   # list[str]

Semantics of class c:
    e_k = L2norm( text_proj( text_encoder([BOS, prompts_k, EOS]) ) )    for each k
    T_c = L2norm( mean_k(e_k) )                                          # ensemble

Covered global classes (33):
    Anatomy (cholecystectomy): cystic_plate, calot_triangle, cystic_artery,
        cystic_duct, gallbladder, tool, abdominal_wall, liver,
        gastrointestinal_tract, fat, grasper, connective_tissue, blood,
        l_hook_electrocautery, hepatic_vein, liver_ligament
    Robotic (endovis2018 + 2017_parts): shaft, clasper, wrist,
        kidney_parenchyma, covered_kidney, thread, clamps, suturing_needle,
        suction_instrument, small_intestine, ultrasound_probe
    Instrument types (endovis2017_type): bipolar_forceps, prograsp_forceps,
        large_needle_driver, vessel_sealer, grasping_retractor,
        monopolar_curved_scissors
"""
from __future__ import annotations
from typing import Dict, List


TEMPLATES: Dict[str, List[str]] = {
    # -------- cholecystectomy anatomy --------
    'cystic_plate': [
        'cystic plate, a thin connective tissue layer on the gallbladder bed',
        'fibrous cystic plate beneath the gallbladder in laparoscopic view',
        'a pale connective layer attached to the liver bed where the gallbladder sits',
        'cystic plate exposed during gallbladder dissection',
    ],
    'calot_triangle': [
        "Calot's triangle, the hepatocystic triangle formed by cystic duct, common hepatic duct and liver edge",
        'the surgical triangle of Calot containing the cystic artery and cystic duct',
        'hepatocystic triangle exposed for critical view of safety',
        'anatomical region between cystic duct, hepatic duct, and inferior liver edge',
    ],
    'cystic_artery': [
        'cystic artery, a small red branching vessel supplying the gallbladder',
        'a thin pulsatile artery running to the gallbladder neck',
        'cystic artery visible within Calot triangle',
        'small arterial branch from the right hepatic artery to the gallbladder',
    ],
    'cystic_duct': [
        'cystic duct, a narrow tubular duct connecting the gallbladder to the common bile duct',
        'a pale tubular biliary structure draining the gallbladder',
        'cystic duct dissected and clipped during cholecystectomy',
        'small bile duct emerging from the gallbladder neck',
    ],
    'gallbladder': [
        'gallbladder, a pear-shaped greenish-yellow sac on the underside of the liver',
        'a distended bile-filled gallbladder being retracted upward',
        'gallbladder fundus and body visible in laparoscopic cholecystectomy',
        'green pear-shaped organ attached to the inferior surface of the liver',
    ],
    'tool': [
        'a surgical instrument visible in the laparoscopic field',
        'a grasper or dissector being used during laparoscopic surgery',
        'a metallic surgical tool tip introduced through a trocar',
        'an instrument shaft and jaws performing tissue manipulation',
    ],
    'abdominal_wall': [
        'the inner surface of the abdominal wall, showing muscle and peritoneum',
        'pale-pink muscular layer of the anterior abdominal wall seen laparoscopically',
        'parietal peritoneum lining the abdominal cavity',
        'abdominal wall tissue with visible trocar entry site',
    ],
    'liver': [
        'liver, a large reddish-brown smooth-surfaced organ',
        'right and left lobes of the liver dominating the upper abdomen',
        'glossy hepatic surface adjacent to the gallbladder',
        'soft brown hepatic parenchyma with rounded edge',
    ],
    'gastrointestinal_tract': [
        'a section of the gastrointestinal tract, pink coiled tubular bowel',
        'small bowel loops or stomach wall visible in the abdominal cavity',
        'intestinal serosa with mesenteric attachments',
        'tubular bowel structure with peristaltic appearance',
    ],
    'fat': [
        'yellow adipose tissue surrounding abdominal organs',
        'lobulated fat deposits on the omentum or mesentery',
        'fatty tissue covering the gallbladder or near Calot triangle',
        'soft globular yellow fat between organs',
    ],
    'grasper': [
        'a laparoscopic grasper instrument with two articulating jaws',
        'metallic grasping forceps holding tissue in the surgical field',
        'a thin grasper introduced through a trocar to retract the gallbladder',
        'instrument jaws clamped on tissue during dissection',
    ],
    'connective_tissue': [
        'whitish fibrous connective tissue between organs',
        'areolar connective tissue dissected in the surgical plane',
        'fibrous strands attaching the gallbladder to the liver bed',
        'pale connective layers being separated during dissection',
    ],
    'blood': [
        'fresh red blood pooling in the surgical field',
        'bleeding from a dissected vessel during cholecystectomy',
        'bright red blood obscuring tissue surfaces',
        'a small hemorrhage being irrigated and suctioned',
    ],
    'l_hook_electrocautery': [
        'L-hook electrocautery probe, a thin metal hook for tissue dissection',
        'monopolar L-shaped electrocautery hook performing dissection',
        'a hooked electrocautery instrument cutting cystic duct attachments',
        'energy device with L-shaped tip used for fine dissection',
    ],
    'hepatic_vein': [
        'hepatic vein, a bluish vessel draining liver tissue',
        'a deep blue venous structure within the hepatic parenchyma',
        'hepatic venous tributary visible during liver dissection',
        'large dark-blue vein returning blood from the liver',
    ],
    'liver_ligament': [
        'falciform ligament of the liver, a thin white fibrous fold',
        'pale ligamentous attachment of the liver to the abdominal wall',
        'liver ligament stretching between liver lobes',
        'thin avascular fibrous ligament associated with the liver',
    ],

    # -------- robotic surgery (endovis2018 + 2017_parts) --------
    'shaft': [
        'the long cylindrical metallic shaft of a robotic surgical instrument',
        'instrument shaft, the elongated rod connecting tool tip to actuator',
        'straight metallic instrument body inserted through a trocar',
        'silver shaft of a da Vinci robotic instrument',
    ],
    'clasper': [
        'the clasper tip of a robotic instrument, the working jaws',
        'instrument clasper, the gripping or cutting tip at the end of the wrist',
        'distal jaws of a robotic tool holding or cutting tissue',
        'tip of a robotic instrument with two articulating prongs',
    ],
    'wrist': [
        'the articulating wrist joint of a robotic instrument',
        'instrument wrist, the flexible joint between shaft and clasper',
        'multi-axis robotic wrist providing tool tip dexterity',
        'pivot joint of a da Vinci instrument allowing angular motion',
    ],
    'kidney_parenchyma': [
        'kidney parenchyma, the reddish-brown renal tissue',
        'exposed renal cortex with smooth glistening surface',
        'soft hemorrhagic-prone kidney tissue during partial nephrectomy',
        'renal parenchyma cut surface showing pale cortex and dark medulla',
    ],
    'covered_kidney': [
        'kidney covered with perirenal fat and Gerota fascia',
        'kidney still encased in surrounding adipose and fibrous capsule',
        'unmobilised kidney covered by retroperitoneal tissue',
        'covered renal surface with overlying fat before dissection',
    ],
    'thread': [
        'a surgical suture thread, thin and usually dark or blue',
        'a single strand of suture material in the surgical field',
        'monofilament thread used for tissue approximation',
        'dark suture thread emerging from a tissue site',
    ],
    'clamps': [
        'a surgical clamp grasping tissue or a vessel',
        'metallic clamp applied for hemostasis during dissection',
        'two locking jaws of a clamp holding a structure closed',
        'hemostatic clamp visible in the operative field',
    ],
    'suturing_needle': [
        'a small curved metallic suturing needle',
        'a crescent-shaped silver needle held by a needle driver',
        'surgical needle threading through tissue for closure',
        'half-circle suture needle in the operative view',
    ],
    'suction_instrument': [
        'a suction-irrigation instrument, a tubular tool for evacuating fluid',
        'open-ended suction probe removing blood or irrigation from the field',
        'long thin suction device clearing the surgical site',
        'irrigation suction cannula inserted into the abdomen',
    ],
    'small_intestine': [
        'small intestine, a pink coiled tubular bowel structure',
        'loops of small bowel within the abdominal cavity',
        'glistening serosal surface of the small intestine',
        'long convoluted segment of small bowel with mesentery',
    ],
    'ultrasound_probe': [
        'a laparoscopic ultrasound probe, a rectangular imaging head on a flexible shaft',
        'intraoperative ultrasound transducer placed on tissue surface',
        'rectangular probe head of an ultrasound device for tissue imaging',
        'flexible ultrasound probe scanning the kidney or liver surface',
    ],

    # -------- instrument types (endovis2017_type) --------
    'bipolar_forceps': [
        'bipolar forceps, a two-pronged electrocautery instrument for sealing tissue',
        'da Vinci bipolar grasper with insulated jaws for coagulation',
        'twin-tip instrument delivering bipolar energy to tissue',
        'bipolar cauterization forceps with two parallel prongs',
    ],
    'prograsp_forceps': [
        'ProGrasp forceps, a heavy-duty ratcheted grasping instrument',
        'large jaw robotic grasper with serrated tips for firm tissue retraction',
        'wide-mouth grasping forceps with a locking grip',
        'broad-tip grasping instrument used for traction',
    ],
    'large_needle_driver': [
        'large needle driver, a robust instrument designed to grip suture needles',
        'heavy needle holder with textured jaws for needle manipulation',
        'robotic needle driver holding a surgical needle during suturing',
        'large-jaw needle driver instrument in a robotic surgery scene',
    ],
    'vessel_sealer': [
        'vessel sealer, a wide-jawed energy device for sealing and cutting vessels',
        'bipolar advanced energy device that fuses and divides blood vessels',
        'broad-jaw sealing instrument used on vascular pedicles',
        'tissue sealing device with integrated cutting blade',
    ],
    'grasping_retractor': [
        'grasping retractor, a long instrument for grasping and holding tissue aside',
        'fan or paddle-shaped retractor combined with a grasper',
        'instrument holding tissue out of the surgical field for exposure',
        'long retractor with grasping tip for organ mobilisation',
    ],
    'monopolar_curved_scissors': [
        'monopolar curved scissors, a curved cutting instrument with electrocautery capability',
        'hot scissors with a slight curvature used for fine dissection',
        'curved scissor blades delivering monopolar energy for cutting',
        'robotic curved scissors performing sharp dissection',
    ],
}


# Sanity check on import
def _check_coverage():
    EXPECTED = {
        'cystic_plate', 'calot_triangle', 'cystic_artery', 'cystic_duct', 'gallbladder', 'tool',
        'abdominal_wall', 'liver', 'gastrointestinal_tract', 'fat', 'grasper', 'connective_tissue',
        'blood', 'l_hook_electrocautery', 'hepatic_vein', 'liver_ligament',
        'shaft', 'clasper', 'wrist', 'kidney_parenchyma', 'covered_kidney',
        'thread', 'clamps', 'suturing_needle', 'suction_instrument', 'small_intestine',
        'ultrasound_probe',
        'bipolar_forceps', 'prograsp_forceps', 'large_needle_driver', 'vessel_sealer',
        'grasping_retractor', 'monopolar_curved_scissors',
    }
    missing = EXPECTED - set(TEMPLATES.keys())
    extra = set(TEMPLATES.keys()) - EXPECTED
    if missing or extra:
        raise RuntimeError(f'TEMPLATES mismatch — missing={missing} extra={extra}')

_check_coverage()


if __name__ == '__main__':
    print(f'{len(TEMPLATES)} classes, '
          f'{sum(len(v) for v in TEMPLATES.values())} templates total '
          f'({sum(len(v) for v in TEMPLATES.values())/len(TEMPLATES):.1f}/class)')
