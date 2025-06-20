PALETTE = {
    0: [255, 255, 255], # white  -  background
    1: [204, 50, 50],   # red    -  old
    2: [231, 180, 22],  # yellow -  update
    3: [45, 201, 55]    # green  -  new
}

QUAD_WEIGHTS = {
    0: 0, # background
    1: 0.1,   # old
    2: 0.5,  # update
    3: 1    # new
}

VIEWPOINTS = {
    1: {
        "azim": [
            0
        ],
        "elev": [
            0
        ],
        "sector": [
            "front"
        ]
    },
    2: {
        "azim": [
            0,
            30
        ],
        "elev": [
            0,
            0
        ],
        "sector": [
            "front",
            "front"
        ]
    },
    'objaverse22': {
        "azim": [
            0,
            45,
            45
        ],
        "elev": [
            0,
            0,
            35
        ],
        "sector": [
            "front",
            "right",
            "front right"
        ]
    },
    4: {
        "azim": [
            45,
            315,
            135,
            225,
        ],
        "elev": [
            0,
            0,
            0,
            0,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",
        ]
    },
    6: {
        "azim": [
            0,
            90,
            270,
            0,
            180,
            0
        ],
        "elev": [
            0,
            0,
            0,
            90,
            0,
            -90
        ],
        "sector": [
            "front",
            "right",
            "left",
            "top",
            "back",
            "bottom",
        ]
    },
    "shapenet": {
        "azim": [
            270,
            315,
            225,
            0,
            180,
            45,
            135,
            90,
            270,
            270
        ],
        "elev": [
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            90,
            -90
        ],
        "sector": [
            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",
            "top",
            "bottom",
        ]
    },
    "objaverse": {
        "azim": [
            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,
            0,
            0
        ],
        "elev": [
            0,
            35,
            35,
            0,
            0,
            35,
            35,
            0,
            90,
            -90
        ],
        "sector": [
            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",
            "top",
            "bottom",
        ]
    },
    12: {
        "azim": [
            45,
            315,
            135,
            225,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,
        ],
        "elev": [
            0,
            0,
            0,
            0,

            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",
        ]
    },
    20: {
        "azim": [
            45,
            315,
            135,
            225,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,
        ],
        "elev": [
            0,
            0,
            0,
            0,

            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,

            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",
        ]
    },
    36: {
        "azim": [
            45,
            315,
            135,
            225,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,
        ],
        "elev": [
            0,
            0,
            0,
            0,

            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,

            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,

            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,

            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",

            "top front",
            "top right",
            "top left",
            "top right",
            "top left",
            "top right",
            "top left",
            "top back",

            "front right",
            "front left",
            "front right",
            "front left",
            "back right",
            "back left",
            "back right",
            "back left",

            "front right",
            "front left",
            "front right",
            "front left",
            "back right",
            "back left",
            "back right",
            "back left",
        ]
    },
    68: {
        "azim": [
            45,
            315,
            135,
            225,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            0,
            45,
            315,
            90,
            270,
            135,
            225,
            180,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5,

            22.5,
            337.5,
            67.5,
            292.5,
            112.5,
            247.5,
            157.5,
            202.5
        ],
        "elev": [
            0,
            0,
            0,
            0,

            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,

            60,
            60,
            60,
            60,
            60,
            60,
            60,
            60,

            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,

            45,
            45,
            45,
            45,
            45,
            45,
            45,
            45,

            -30,
            -30,
            -30,
            -30,
            -30,
            -30,
            -30,
            -30,

            -60,
            -60,
            -60,
            -60,
            -60,
            -60,
            -60,
            -60,

            -15,
            -15,
            -15,
            -15,
            -15,
            -15,
            -15,
            -15,

            -45,
            -45,
            -45,
            -45,
            -45,
            -45,
            -45,
            -45,
        ],
        "sector": [
            "front right",
            "front left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",

            "top front",
            "top right",
            "top left",
            "top right",
            "top left",
            "top right",
            "top left",
            "top back",

            "front right",
            "front left",
            "front right",
            "front left",
            "back right",
            "back left",
            "back right",
            "back left",

            "front right",
            "front left",
            "front right",
            "front left",
            "back right",
            "back left",
            "back right",
            "back left",

            "front",
            "front right",
            "front left",
            "right",
            "left",
            "back right",
            "back left",
            "back",

            "bottom front",
            "bottom right",
            "bottom left",
            "bottom right",
            "bottom left",
            "bottom right",
            "bottom left",
            "bottom back",

            "bottom front right",
            "bottom front left",
            "bottom front right",
            "bottom front left",
            "bottom back right",
            "bottom back left",
            "bottom back right",
            "bottom back left",

            "bottom front right",
            "bottom front left",
            "bottom front right",
            "bottom front left",
            "bottom back right",
            "bottom back left",
            "bottom back right",
            "bottom back left",
        ]
    }
}