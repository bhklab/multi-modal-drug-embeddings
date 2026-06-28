from typing import Dict



def unpack_parameters(
    D:Dict,
    return_dict:bool=False
    ):
	
	if len(D.values())>1:
		return D if return_dict else tuple(D.values())
	else:
		return D if return_dict else tuple(D.values())[0]



KEEP_MOAS = ["DNA inhibitor", #X
	"Epidermal growth factor receptor erbB1 inhibitor", #X
"Tubulin inhibitor", #X
"MAP kinase p38 alpha inhibitor", #X
"Tyrosine-protein kinase receptor FLT3 inhibitor",# X
"Fibroblast growth factor receptor inhibitor", #X
"Estrogen receptor modulator", #X
"Tyrosine-protein kinase BTK inhibitor",#x
"ALK tyrosine kinase receptor inhibitor",#x
"Tyrosine-protein kinase JAK2 inhibitor",#
"PARP 1, 2 and 3 inhibitor", #X
"DNA disrupting agent",#X
"Tubulin inhibitor",#X
#"26S proteasome inhibitor", #X
"Histone-lysine N-methyltransferase EZH2 inhibitor",#X
"Tyrosine-protein kinase ABL inhibitor",#X
"Serine/threonine-protein kinase mTOR inhibitor",# X
"PI3-kinase class I inhibitor",#X
"Serine/threonine-protein kinase AKT inhibitor",#X
"Tyrosine-protein kinase receptor FLT3 inhibitor",#X
"Tyrosine-protein kinase JAK1 inhibitor",#X
"Cyclin-dependent kinase 1 inhibitor",#X
"Cyclin-dependent kinase 2 inhibitor",#X
"Cyclin-dependent kinase inhibitor",#X
"Receptor protein-tyrosine kinase erbB-2 inhibitor",#X
"Tyrosine-protein kinase JAK2 inhibitor",#X
"PI3-kinase p110-delta subunit inhibitor", #X
"Serine/threonine-protein kinase PLK1 inhibitor",# X
"Tyrosine-protein kinase SYK inhibitor" #X
] 



MOA_RENAME = {
	"Cyclin-dependent kinase 1 inhibitor": 'CDK Inhibitor',
	"Cyclin-dependent kinase 2 inhibitor" : 'CDK Inhibitor',
	"Cyclin-dependent kinase inhibitor":  'CDK Inhibitor',
	"Receptor protein-tyrosine kinase erbB-2 inhibitor": "ERBB-2 Inhibitor",
	"Tyrosine-protein kinase JAK2 inhibitor": "JAK Inhibitor",
	"Tyrosine-protein kinase JAK1 inhibitor": "JAK Inhibitor",
	"Fibroblast growth factor receptor inhibitor": "FGFR Inhibitor",
	"Tyrosine-protein kinase receptor FLT3 inhibitor": "FLT3 Inhibitor",
	"Epidermal growth factor receptor erbB1 inhibitor": "EGFR Inhibitor",
	"MAP kinase p38 alpha inhibitor": "MAPK14 Inhibitor",
	"PI3-kinase p110-delta subunit inhibitor": 'PI3K Inhibitor',
	"Tyrosine-protein kinase SYK inhibitor": "SYK Inhibitor",
	"Serine/threonine-protein kinase PLK1 inhibitor": "PLK1 Inhibitor",
	"Tyrosine-protein kinase BTK inhibitor": "BTK Inhibitor",
	"ALK tyrosine kinase receptor inhibitor": "ALK Inhibitor",
	"Tyrosine-protein kinase ABL inhibitor": "ABL Inhibitor",
	"Serine/threonine-protein kinase mTOR inhibitor": "mTOR Inhibitor",
	"PI3-kinase p110-delta subunit inhibitor": "PI3K Inhibitor",
	"Serine/threonine-protein kinase PLK1 inhibitor": "PLK1 Inhibitor",
	"Receptor protein-tyrosine kinase erbB-2 inhibitor":"ERBB2 Inhibitor",
	"Serine/threonine-protein kinase AKT inhibitor":"AKT Inhibitor",
	"PI3-kinase class I inhibitor": 'PI3K Inhibitor',
	"Histone-lysine N-methyltransferase EZH2 inhibitor": "EZH2"
}

CATEGORY_RULES = {
    "Receptor Targeted": [
        "receptor agonist",
        "receptor antagonist",
        "receptor modulator",
        "receptor inverse agonist",
        "receptor partial agonist",
        "positive allosteric modulator",
        "negative allosteric modulator",
        "allosteric antagonist",
    ],

    "Ion Channels & Transporters": [
        "channel blocker",
        "channel opener",
        "transporter inhibitor",
        "transporter substrate",
        "transporter releasing agent",
        "cotransporter inhibitor",
        "ATPase inhibitor",
        "ENaC blocker",
        "voltage-gated",
        "potassium channel",
        "sodium channel",
        "calcium channel",
    ],

    "Enzymes": [
        "enzyme inhibitor",
        "reductase inhibitor",
        "synthetase inhibitor",
        "synthase inhibitor",
        "hydroxylase inhibitor",
        "dehydrogenase inhibitor",
        "oxidase inhibitor",
        "polymerase inhibitor",
        "protease inhibitor",
        "phosphodiesterase",
        "cyclooxygenase",
        "lipoxygenase",
        "acetylcholinesterase",
        "monoamine oxidase",
        "topoisomerase",
        "kinase inhibitor",
        "transferase inhibitor",
        "farnesyltransferase inhibitor",
    ],

    "Kinases & Signaling": [
        "JAK",
        "RAF",
        "AKT",
        "FLT3",
        "BTK",
        "SYK",
        "SRC",
        "ABL",
        "PI3-kinase",
        "protein kinase",
        "growth factor receptor inhibitor",
    ],

    "Nuclear Receptors & Hormonal": [
        "estrogen receptor",
        "androgen receptor",
        "progesterone receptor",
        "glucocorticoid receptor",
        "mineralocorticoid receptor",
        "thyroid hormone receptor",
        "vitamin D receptor",
        "retinoic acid receptor",
        "retinoid receptor",
        "FXR agonist",
        "PPAR",
    ],

    "Antibacterial / Antifungal": [
        "penicillin-binding protein",
        "beta-lactamase",
        "DNA gyrase",
        "topoisomerase IV",
        "70S ribosome",
        "80S Ribosome",
        "cell wall",
        "bacterial",
        "lanosterol",
        "ergosterol",
        "dihydrofolate reductase",
        "dihydropteroate",
        "RNA polymerase inhibitor",
    ],

    "Antiviral": [
        "human immunodeficiency virus",
        "hepatitis C virus",
        "neuraminidase inhibitor",
        "reverse transcriptase",
        "integrase inhibitor",
        "DNA polymerase inhibitor",
        "RNA-directed RNA polymerase",
        "nonstructural protein",
        "envelope glycoprotein",
        "polymerase acidic protein",
        "DNA terminase",
    ],

    "Oncology / Cytotoxic": [
        "cytotoxic",
        "DNA disrupting agent",
        "tubulin inhibitor",
        "proteasome inhibitor",
        "PARP",
        "EZH2 inhibitor",
        "Bcl-2 inhibitor",
        "CDK",
        "exportin-1 inhibitor",
    ],

    "Neuropharmacology": [
        "dopamine",
        "serotonin",
        "GABA",
        "opioid",
        "adrenergic",
        "acetylcholine",
        "muscarinic",
        "nicotinic",
        "glutamate",
        "NMDA",
        "glycine receptor",
        "melatonin receptor",
        "neurokinin",
    ],

    "Cardiovascular & Renal": [
        "angiotensin",
        "renin inhibitor",
        "endothelin",
        "thrombin inhibitor",
        "factor X inhibitor",
        "P2Y12",
        "vasopressin",
        "guanylate cyclase",
        "prostanoid",
    ],

    "Metabolic & Endocrine": [
        "SGLT",
        "sodium/glucose cotransporter",
        "DPP",
        "sulfonylurea",
        "lipase inhibitor",
        "cholesterol",
        "HMG-CoA",
        "thyroid",
        "glucose",
    ],

    "Chelators / Supplements / Supportive": [
        "chelating agent",
        "supplement",
        "antioxidant",
        "surfactant",
        "excipient",
        "laxative",
        "osmotic agent",
        "diagnostic",
        "diagnostic agent",
        "reducing agent",
    ]
}
