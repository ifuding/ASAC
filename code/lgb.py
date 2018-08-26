import lightgbm as lgb
import pandas as pd
import keras_train
import numpy as np
# import config


def lgbm_train(train_part, train_part_label, valide_part, valide_part_label, fold_seed,
        fold = 5, train_weight = None, valide_weight = None, flags = None):
    """
    LGBM Training
    """
    CATEGORY_FEATURES = keras_train.USED_FEATURE_LIST
    FEATURE_LIST = keras_train.USED_FEATURE_LIST
    if flags.stacking:
        FEATURE_LIST += ['emb_' + str(i) for i in range(len(CATEGORY_FEATURES) * 5)] + ['k_pred']
    print("-----LGBM training-----")

    d_train = lgb.Dataset(train_part[FEATURE_LIST].values, train_part_label, weight = train_weight, 
            feature_name = FEATURE_LIST) #, categorical_feature = CATEGORY_FEATURES) #, init_score = train_part[:, -1])
    d_valide = lgb.Dataset(valide_part[FEATURE_LIST].values, valide_part_label, weight = valide_weight,
            feature_name = FEATURE_LIST) #, categorical_feature = CATEGORY_FEATURES) #, init_score = valide_part[:, -1])
    params = {
    # 'num_leaves':-1,
        'task': 'train',
        'min_sum_hessian_in_leaf':None,
        'max_depth':10,
        'learning_rate':0.005,
        'feature_fraction':0.8,
        'verbose':-1,
        'objective': 'multiclass',
        'num_class':6,
        'metric': 'multi_logloss',
        'num_boost_round':3000,
        'drop_rate':None,
        'bagging_fraction':0.6,
        'bagging_freq':5,
        'early_stopping_round':100,
        # 'min_data_in_leaf':100,
        'max_bin': None,
        'scale_pos_weight':None,
    }
    # params.update(config.all_params)
    print ("lightgbm params: {0}\n".format(params))

    bst = lgb.train(
                    params ,
                    d_train,
                    verbose_eval = 200,
                    valid_sets = [d_train, d_valide],
                    # feature_name= keras_train.DENSE_FEATURE_LIST,
                    #feval = gini_lgbm
                    #num_boost_round = 1
                    )
    #pred = model_eval(bst, 'l', valide_part)
    #print(pred[:10])
    #print(valide_part_label[:10])
    #print(valide_part[:10, -1])
    # exit(0)
    feature_imp = bst.feature_importance(importance_type = 'gain')
    sort_ind = np.argsort(feature_imp)[::-1]
    print (np.c_[np.array(FEATURE_LIST)[sort_ind], feature_imp[sort_ind]][:10])
    # print (np.array(keras_train.FEATURE_LIST)[np.argsort(feature_imp)])
    # exit(0)
    # cv_result = lgb.cv(params, d_train, nfold=fold) #, feval = gini_lgbm)
    # pd.DataFrame(cv_result).to_csv('cv_result', index = False)
    # exit(0)
    return bst