MODEL_PARAMS = [
    {
        'model': 'LightGBM',
        'params': {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': ['l2', 'l1'],
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        },
        'exogs': True
    },
    {
        'model': 'LightGBM',
        'params': {
            'objective': 'regression',
            'verbose': -1
        },
        'exogs': True
    },
    {
        'model': 'LightGBM',
        'params': {
            "objective" : "poisson",
            "metric" :"rmse",
            "force_row_wise" : True,
            "learning_rate" : 0.075,
            "sub_row" : 0.75,
            "bagging_freq" : 1,
            "lambda_l2" : 0.1,
            "metric": ["rmse"],
            'verbose': -1,
            'num_iterations' : 1200,
            'num_leaves': 128,
            "min_data_in_leaf": 100,
        },
        'exogs': False
    },
    {
        'model': 'LightGBM',
        'params': {
            'boosting_type': 'gbdt',
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'rmse',
            'subsample': 0.5,
            'subsample_freq': 1,
            'learning_rate': 0.03,
            'num_leaves': 2 ** 11 - 1,
            'min_data_in_leaf': 2 ** 12 - 1,
            'feature_fraction': 0.5,
            'max_bin': 100,
            'n_estimators': 1, #1400
            'boost_from_average': False,
            'verbose': -1
        },
        'exogs': True
    },
    {
        'model': 'Exponential Smoothing',
        'params': None,
        'exogs': False
    },
    {
        'model': 'Moving Average',
        'params': None,
        'exogs': False
    },
    {
        'model': 'ARIMA',
        'params': None,
        'exogs': True
    },
    # {
    #     'model': 'Prophet',
    #     'params': None,
    #     'exogs': False
    # }
]

FEATURE_SETS = [
    [],
    [
        "sintomas covid"
    ],
    [
        "sintomas covid",
        "trabajos"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento",
        "real estate"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento",
        "real estate",
        "casas"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento",
        "real estate",
        "casas",
        "tenis"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento",
        "real estate",
        "casas",
        "tenis",
        "botas"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento",
        "real estate",
        "casas",
        "tenis",
        "botas",
        "sandalias"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento",
        "real estate",
        "casas",
        "tenis",
        "botas",
        "sandalias",
        "adidas"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento",
        "real estate",
        "casas",
        "tenis",
        "botas",
        "sandalias",
        "adidas",
        "nike"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento",
        "real estate",
        "casas",
        "tenis",
        "botas",
        "sandalias",
        "adidas",
        "nike",
        "converse"
    ],
    [
        "sintomas covid",
        "trabajos",
        "carros",
        "reclutamiento",
        "real estate",
        "casas",
        "tenis",
        "botas",
        "sandalias",
        "adidas",
        "nike",
        "converse",
        "desempleo"
    ]
]