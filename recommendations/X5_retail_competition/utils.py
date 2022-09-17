import numpy as np
import pandas as pd


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)

    return precision


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)

    if sum(flags) == 0:
        return 0

    sum_ = 0
    for i in range(k):

        if flags[i]:
            p_k = precision_at_k(recommended_list, bought_list, k=i + 1)
            sum_ += p_k

    result = sum_ / sum(flags)

    return result


def map_k(recommend_list, bought_list, k=5):
    return np.mean([ap_k(rec, bt, k) for rec, bt in zip(recommend_list, bought_list)])


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def calc_precision_at_k(df_data, top_k):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: precision_at_k(row[col_name], row['actual'], k=top_k), axis=1).mean()


def calc_recall(df_data, top_k):
    for col_name in df_data.columns[2:]:
        yield col_name, df_data.apply(lambda row: recall_at_k(row[col_name], row['actual'], k=top_k), axis=1).mean()


def calc_map_at_k(df_data):
    for col_name in df_data.columns[2:]:
        yield col_name, map_k(df_data[col_name].values.tolist(), df_data['actual'].values.tolist())


def reduce_memory(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)[:4] != 'uint' and str(col_type) != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif str(col_type)[:4] != 'uint':
            df[col] = df[col].astype('category')
    return df


def transform_data_for_eval(dataset, rec_col, user_col='user_id'):
    eval_dataset = dataset[[user_col, rec_col]].copy()
    eval_dataset[rec_col] = eval_dataset[rec_col].apply(lambda x: ' '.join([str(i) for i in x]))
    eval_dataset.rename(columns={
        user_col: 'UserId',
        rec_col: 'Predicted'
    }, inplace=True)
    return eval_dataset


def prefilter_items(data, take_n_popular=5000, item_features=None, n_weeks=95):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(
            item_features.groupby('department')['item_id'].nunique().sort_values(ascending=False)).reset_index()
        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 50]

    # уберем товары, не продававшиеся более n_week недель
    data = data[data['week_no'] >= data['week_no'].max() - n_weeks]

    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-N, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


def similar_products(v, n=5):
    ms = model.wv.similar_by_vector(v, topn=n)
    new_ms = [item[0] for item in ms]
    return new_ms


def aggregate_vectors(items, agg_func=np.mean):
    n = len(items)
    item_vec = np.zeros((n, 100))
    for i in range(n):
        try:
            item_vec[i, :] = model.wv.get_vector(items[i], norm=True)
        except KeyError:
            continue

    return agg_func(item_vec, axis=0)


top_purchases = data_train.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()


def get_self_top_purchases(user_id, N=5):
    return top_purchases[top_purchases.user_id == user_id].item_id.head(N).tolist()


def popularity_recommendation(data, n=5):
    """Топ-n популярных товаров"""

    popular = data.groupby('item_id')['sales_value'].sum().reset_index()
    popular.sort_values('sales_value', ascending=False, inplace=True)

    recs = popular.head(n).item_id

    return recs.tolist()


train_ids = data_train["user_id"].unique()


def get_prediction(user_id, N=5, agg_func=np.mean):
    if user_id in train_ids:
        purchases = get_self_top_purchases(user_id, N + 1)
        agg_vector = aggregate_vectors(purchases, agg_func=agg_func)
        recs = similar_products(agg_vector, N + 1)
    else:
        recs = popularity_recommendation(data_train, N + 1)

    if "999999" in recs:
        recs.remove("999999")
    else:
        recs = recs[:-1]
    return recs
