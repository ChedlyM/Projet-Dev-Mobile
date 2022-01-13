from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd 
import numpy as np
from flask import Blueprint, jsonify, request,Flask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#driver = GraphDatabase.driver( "bolt://localhost:7687",auth=basic_auth("neo4j", "123456"))

main = Blueprint('main',__name__)

@main.route('/recomm',methods=['POST'])
def recomm():
    df1=pd.read_csv('./db/food.csv')
    df1.columns = ['food_id','title','canteen_id','price', 'num_orders', 'category', 'avg_rating', 'num_rating', 'tags']
    # mean of average ratings of all items
    C= df1['avg_rating'].mean()

    # the minimum number of votes required to appear in recommendation list, i.e, 60th percentile among 'num_rating'
    m= df1['num_rating'].quantile(0.6)

    # items that qualify the criteria of minimum num of votes
    q_items = df1.copy().loc[df1['num_rating'] >= m]

    # Calculation of weighted rating based on the IMDB formula
    def weighted_rating(x, m=m, C=C):
        v = x['num_rating']
        R = x['avg_rating']
        return (v/(v+m) * R) + (m/(m+v) * C)

    # Applying weighted_rating to qualified items
    q_items['score'] = q_items.apply(weighted_rating, axis=1)

    # Shortlisting the top rated items and popular items
    top_rated_items = q_items.sort_values('score', ascending=False)
    pop_items= df1.sort_values('num_orders', ascending=False)
    def create_soup(x):            
        tags = x['tags'].lower().split(', ')
        tags.extend(x['title'].lower().split())
        tags.extend(x['category'].lower().split())
        return " ".join(sorted(set(tags), key=tags.index))
    df1['soup'] = df1.apply(create_soup, axis=1)
    # Import CountVectorizer and create the count matrix
    from sklearn.feature_extraction.text import CountVectorizer
    count = CountVectorizer(stop_words='english')

    # df1['soup']
    count_matrix = count.fit_transform(df1['soup'])

    # Compute the Cosine Similarity matrix based on the count_matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices_from_title = pd.Series(df1.index, index=df1['title'])
    indices_from_food_id = pd.Series(df1.index, index=df1['food_id'])
    # Function that takes in food title or food id as input and outputs most similar dishes 
    def get_recommendations(title="", cosine_sim=cosine_sim, idx=-1):
        # Get the index of the item that matches the title
        if idx == -1 and title != "":
            idx = indices_from_title[title]

        # Get the pairwsie similarity scores of all dishes with that dish
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the dishes based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the 10 most similar dishes
        sim_scores = sim_scores[1:3]

        # Get the food indices
        food_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar dishes
        return food_indices
    # fetch few past orders of a user, based on which personalized recommendations are to be made
    def get_latest_user_orders(user_id, orders, num_orders=3):
        counter = num_orders
        order_indices = []
        
        for index, row in orders[['user_id']].iterrows():
            if row.user_id == user_id:
                counter = counter -1
                order_indices.append(index)
            if counter == 0:
                break
                
        return order_indices

    # utility function that returns a DataFrame given the food_indices to be recommended
    def get_recomms_df(food_indices, df1, columns, comment):
        row = 0
        df = pd.DataFrame(columns=columns)
        
        for i in food_indices:
            df.loc[row] = df1[['title', 'canteen_id', 'price']].loc[i]
            df.loc[row].comment = comment
            row = row+1
        return df

    # return food_indices for accomplishing personalized recommendation using Count Vectorizer
    def personalised_recomms(orders, df1, user_id, columns, comment="based on your past orders"):
        order_indices = get_latest_user_orders(user_id, orders)
        food_ids = []
        food_indices = []
        recomm_indices = []
        
        for i in order_indices:
            food_ids.append(orders.loc[i].food_id)
        for i in food_ids:
            food_indices.append(indices_from_food_id[i])
        for i in food_indices:
            recomm_indices.extend(get_recommendations(idx=i))
            
        return get_recomms_df(set(recomm_indices), df1, columns, comment)

    # Simply fetch new items added by vendor or today's special at home canteen
    def get_new_and_specials_recomms(new_and_specials, users, df1, canteen_id, columns, comment="new/today's special item  in your home canteen"):
        food_indices = []
        
        for index, row in new_and_specials[['canteen_id']].iterrows():
            if row.canteen_id == canteen_id:
                food_indices.append(indices_from_food_id[new_and_specials.loc[index].food_id])
                
        return get_recomms_df(set(food_indices), df1, columns, comment)

    # utility function to get the home canteen given a user id
    def get_user_home_canteen(users, user_id):
        for index, row in users[['user_id']].iterrows():
            if row.user_id == user_id:
                return users.loc[index].home_canteen
        return -1

    # fetch items from previously calculated top_rated_items list
    def get_top_rated_items(top_rated_items, df1, columns, comment="top rated items across canteens"):
        food_indices = []
        
        for index, row in top_rated_items.iterrows():
            food_indices.append(indices_from_food_id[top_rated_items.loc[index].food_id])
            
        return get_recomms_df(food_indices, df1, columns, comment)

    # fetch items from previously calculated pop_items list
    def get_popular_items(pop_items, df1, columns, comment="most popular items across canteens"):
        food_indices = []
        
        for index, row in pop_items.iterrows():
            food_indices.append(indices_from_food_id[pop_items.loc[index].food_id])
            
        return get_recomms_df(food_indices, df1, columns, comment)
    orders = pd.read_csv('./db/orders.csv')
    new_and_specials = pd.read_csv('./db/new_and_specials.csv')
    users = pd.read_csv('./db/users.csv')

    columns = ['title', 'canteen_id', 'price', 'comment']
    current_user = 2
    current_canteen = get_user_home_canteen(users, current_user)

    #By Vendor
    a=get_new_and_specials_recomms(new_and_specials, users, df1, current_canteen, columns)
    #Top rated item
    b=get_top_rated_items(top_rated_items, df1, columns)
    #Most bought item
    c=get_popular_items(pop_items, df1, columns).head(3)
    a_list = a.values.tolist()
    b_list = b.values.tolist()
    c_list = c.values.tolist()
    return jsonify({"By Vendor":a_list,"Top Rated Items":b_list,"Most bought items":c_list})