"""
Shot type prompt templates for product review videos.

Maps product/activity types to optimized prompts for both
image generation (Flux) and video animation (Wan2.1 I2V).
"""

SHOT_PROMPTS = {
    "tshirt": {
        "image": "A young attractive woman wearing a {product_desc}, standing in soft studio lighting, medium shot from waist up, looking at camera with confident smile, Instagram style photo",
        "video": "Woman turns slightly to show her outfit, natural gestures, looking at camera and smiling",
    },
    "shoes": {
        "image": "A young attractive woman standing confidently wearing {product_desc}, full body shot, clean background, fashion photography style, showing the shoes clearly",
        "video": "Woman walks forward a few steps confidently, camera captures her full body and shoes, natural stride",
    },
    "jewelry": {
        "image": "A young attractive woman wearing {product_desc}, close-up portrait, touching the jewelry gently, soft glamour lighting, elegant pose",
        "video": "Woman gently touches her jewelry, tilts head and smiles, natural elegant movement",
    },
    "hat": {
        "image": "A young attractive woman wearing {product_desc}, upper body shot, slightly tilted head, outdoor natural lighting, stylish pose",
        "video": "Woman adjusts her hat, turns head side to side showing the hat, smiles at camera",
    },
    "general_review": {
        "image": "A young attractive woman holding {product_desc} in her hand, medium shot, looking at camera enthusiastically, bright clean background, lifestyle photo",
        "video": "Woman holds up product, gestures while talking about it, animated and enthusiastic expression",
    },
    "cooking": {
        "image": "A young attractive woman in a modern kitchen, cooking at the stove, wearing casual clothes, warm lighting, medium-wide shot",
        "video": "Woman stirs a pot, picks up ingredients, looks at camera and talks, natural cooking movements",
    },
    "outdoor": {
        "image": "A young attractive woman outdoors on a trail in the Southern countryside, casual athletic wear, natural sunlight, medium shot",
        "video": "Woman walks along trail, turns to camera, gestures at surroundings, natural outdoor movement",
    },
    "babysitting": {
        "image": "A young attractive woman in a cozy living room, warm lighting, sitting on couch, friendly and approachable pose, medium shot",
        "video": "Woman sits on couch, gestures while talking, warm and animated expression, cozy home setting",
    },
    "fitness": {
        "image": "A young attractive woman in a gym wearing {product_desc}, athletic pose, bright gym lighting, full body shot",
        "video": "Woman demonstrates a stretch or exercise move, then looks at camera, energetic and confident",
    },
}


def get_image_prompt(shot_type: str, product_desc: str = "the product") -> str:
    """Get the Flux image generation prompt for a shot type."""
    entry = SHOT_PROMPTS.get(shot_type, SHOT_PROMPTS["general_review"])
    return entry["image"].format(product_desc=product_desc)


def get_video_prompt(shot_type: str) -> str:
    """Get the Wan2.1 I2V motion prompt for a shot type."""
    entry = SHOT_PROMPTS.get(shot_type, SHOT_PROMPTS["general_review"])
    return entry["video"]
