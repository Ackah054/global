from .explanations import RegionAttributes

def _severity_from_intensity(mean_intensity, rel_area):
    score = 0.6*mean_intensity + 0.4*min(rel_area*8.0, 1.0)
    if score > 0.75: return "high"
    if score > 0.45: return "moderate"
    return "low"

def _pattern_hint(edge_density, circularity, elongation):
    if edge_density > 0.25 and elongation > 1.6:
        return "linear or wedge-like opacity"
    if circularity > 0.5 and edge_density < 0.15:
        return "rounded focal opacity"
    if edge_density > 0.35 and circularity < 0.3:
        return "irregular patchy opacity"
    return "subtle parenchymal change"

def _body_area(loc_label, task):
    chest_map = {
        "upper-left": "left upper lung zone",
        "upper-center": "upper mediastinal/perihilar zone",
        "upper-right": "right upper lung zone",
        "mid-left": "left mid lung",
        "mid-center": "perihilar region",
        "mid-right": "right mid lung",
        "lower-left": "left lower lung/base",
        "lower-center": "retrocardiac/infrahilar",
        "lower-right": "right lower lung/base",
    }
    brain_map = {
        "upper-left": "left frontal/parietal region",
        "upper-center": "superior midline region",
        "upper-right": "right frontal/parietal region",
        "mid-left": "left temporal/insular region",
        "mid-center": "deep gray/centrum semiovale",
        "mid-right": "right temporal/insular region",
        "lower-left": "left occipital/cerebellar region",
        "lower-center": "posterior fossa/midline",
        "lower-right": "right occipital/cerebellar region",
    }
    return chest_map.get(loc_label, loc_label) if task == "tb" else brain_map.get(loc_label, loc_label)

def _suggest_followup(task, severity):
    if task == "tb":
        if severity == "high":
            return "Recommend sputum test, GeneXpert, and urgent clinical review."
        if severity == "moderate":
            return "Correlate with symptoms and consider sputum testing."
        return "If symptomatic, repeat CXR or consult clinician."
    else:
        if severity == "high":
            return "Urgent neuro review; correlate with diffusion/CTA and NIHSS."
        if severity == "moderate":
            return "Correlate with onset time and consider repeat imaging."
        return "If clinically indicated, repeat scan or observe."

def explain_region(attrs: RegionAttributes, task: str, model_name: str, prob: float):
    severity = _severity_from_intensity(attrs.mean_intensity, attrs.rel_area)
    pattern = _pattern_hint(attrs.edge_density, attrs.circularity, attrs.elongation)
    area = _body_area(attrs.loc_label, task)
    conf_pct = int(round(prob * 100))

    if task == "tb":
        likely = "pulmonary TB-suggestive change"
        nuance = "Consider that overlap with non-TB pneumonias or scarring is possible."
    else:
        likely = "ischemic change"
        nuance = "Note that chronic infarcts or artifacts can mimic acute findings."

    return {
        "title": f"{task.upper()} region ({severity} importance)",
        "summary": f"The model ({model_name}, {conf_pct}% overall) focused on a {pattern} in the {area}.",
        "details": [
            f"Region size: ~{attrs.rel_area*100:.1f}% of image; edge density {attrs.edge_density:.2f}; circularity {attrs.circularity:.2f}.",
            f"Interpreted as {likely} given saliency and morphology.",
            nuance
        ],
        "followup": _suggest_followup(task, severity)
    }
