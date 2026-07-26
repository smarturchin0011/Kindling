import json

ANSWERABLE_GAP = json.dumps(
    {
        "question": "你的受众到底是谁？产品经理和老板需要的东西几乎没有交集。",
        "target_type": "constraint",
        "answerable_from_memory": True,
        "why_critical": "受众不确定，任何框架都是废的",
    },
    ensure_ascii=False,
)

ACTION_GAP = json.dumps(
    {
        "question": "PM 缺这个判断力时，具体做错了什么决定？",
        "target_type": "evidence",
        "answerable_from_memory": False,
        "why_critical": "没有具体事故，框架只能是抽象的",
        "suggested_action": "翻最近的工作记录，找 1 个 PM 因为不懂 AI 做错的决定，写 2 句话",
        "est_minutes": 4,
    },
    ensure_ascii=False,
)

FRAMES = json.dumps(
    {
        "frames": [
            {
                "name": "反直觉清单",
                "thesis": "PM 的错误集中在几个反直觉点上，教学就是逐个拆除",
                "optimizes_for": "立刻可用，每条都能对应真实事故",
                "sacrifices": "不成体系，学完没有全局观",
                "grounded_in_entries": [],
            },
            {
                "name": "概率思维迁移",
                "thesis": "所有错误的根都是确定性思维，只需教一件事",
                "optimizes_for": "极度精简，一个概念贯穿",
                "sacrifices": "抽象，PM 可能听懂但不会用",
                "grounded_in_entries": [],
            },
        ]
    },
    ensure_ascii=False,
)
