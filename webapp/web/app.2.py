from flask import Flask, session, render_template, request, jsonify, redirect
from flask_sqlalchemy import SQLAlchemy
from jinja2 import Template

import sys
# from models import User
import os
import rec_sys
import s3_utils
import json

from engine.dbaccess import DBUtil

app = Flask(__name__, static_url_path='')
app.secret_key = 'xyz'

with open('./data/reclist_json') as json_data:
    bprmodel = json.loads(json.load(json_data))

@app.route('/users/')
def get_users():
  return "get_users"

@app.route('/api/users/<int:u>/')    
def get_user(u):
  # user = User.query.get(u)
  # items = user.reviews
  # result=[]
  # for item in items:
  #   result.append({'asin': item.asin, 'image_url': item.image_url})
  # return jsonify(result)
  pass


# def top_n_rankings(u, n, model_config):
  
  item_bias, user_factors, item_factors = s3_utils.S3Utils.fetch_model_params(model_config)

  recsys = rec_sys.RecSys.factory(item_bias, user_factors, item_factors )

  items = Item.query.limit(Item.query.count()-1)
 
  rankings=[]
  for item in items:
    # i=asin_lut[item.id]
    i=item.id #temp hack until i add the LUT
    rank = recsys.rank(u, i)
    rankings.append({'rank': rank, 'asin': item.asin, 'image_url': item.image_url})
  
  #sort and get top-ten
  rankings = sorted(rankings, key=lambda r: r['rank'], reverse=True)
  rankings = rankings[0:n]
  return rankings

@app.route('/api/users/<int:u>/rankings')
def api_get_rankings(u):
  user = User.query.get(u)
  user_items = user.reviews
  rankings = top_n_rankings(u, 20)
  return jsonify(rankings)
  
@app.route('/users/<int:u>/rankings')
def get_rankings(u):
  user = User.query.get(u)
  user_items = user.reviews
  model_config_id = request.args.get('model_id', '')
  model_config = ModelParamsSet.query.get(model_config_id)

  rankings = top_n_rankings(u, 20, model_config)

  
  return render_template("rankings.html", user=user, rankings=rankings, user_items=user_items, model_config=model_config)
  
# Save new model
@app.route('/models', methods=['POST'])
def prereg():
    # email = None
    # if request.method == 'POST':
    #     email = request.form['email']
    #     # Check that email does not already exist (not a great query, but works)
    #     if not db.session.query(User).filter(User.email == email).count():
    #         reg = User(email)
    #         db.session.add(reg)
    #         db.session.commit()
    #         return render_template('success.html')
    # return render_template('index.html')
    pass


# Save new model
@app.route('/signin', methods=['POST'])
def signin():
    db = DBUtil()
    if request.method == 'POST':
        email = request.form['email']
        print email
        session["email"] = email
        u = db.get_amazon_user(email)
        db.close()
        if len(u) > 0:
            print u
            session['user'] = u
            return redirect("purchase_history.html", code=302)
        else:
            print "no user"
            return redirect("/", code=302)      


@app.route('/tsne')
def tsne():
  return render_template('tsne.1.html')

@app.route('/')
def login():
    print "logging in"
    return render_template('login.html')

@app.route('/recommendation')
def recommendation():
  return render_template('recommendation.html')

@app.route('/get_purchased_items')
def get_purchased_items():
    if "user" not in session.keys():
        return redirect("/", code=302)
    else:
        db = DBUtil()
        u = session['user']
        asin_list = [x[1] for x in u]
        r = db.get_product_item(asin_list)
        dx = [{"asin": x[0], 'title': x[1], 'image_link': x[2], 'price' : float(x[3])} for x in r]
        db.close()
    return jsonify(dx)

@app.route('/get_logged_in_user')
def get_logged_in_user():
    if "user" not in session.keys():
        return jsonify({"user" : ""})
    else:
        email = session['email']
    return jsonify({"user" : email})

# get random products
@app.route('/get_random_products/<int:limit>/')
def get_random_products(limit=10):
    db = DBUtil()
    r = db.get_product_item_random(limit)
#    dx = [{"asin" : x[0], 'title': x[1], 'image_link': x[7]} for x in r]
    dx = [{"asin" : x[0], 'category' : x[1], 'price' : float(x[2]), 'brand' : x[3], 'title': x[4], 'image_link': x[5]} for x in r]
    db.close()
    return jsonify(dx)

# get random category
@app.route('/get_random_category/<int:limit>/')
def get_random_category(limit=16):
    db = DBUtil()
    r = db.get_random_category(limit)
    dx = [{"category": x[0]} for x in r]
    db.close()
    return jsonify(dx)

# get random product by category
@app.route('/get_random_product_by_category', methods=['POST'])
def get_random_product_by_category():
    if not request.json:
        return jsonify("error")
    print request.json
    category = request.json['category']
    limit = request.json['limit']
    # print category
    # print limit

    db = DBUtil()
    r = db.get_random_product_by_category(category, limit)
    dx = [{"asin" : x[0], 'category' : x[1], 'price' : float(x[2]), 'brand' : x[3], 'title': x[4], 'image_link': x[5]} for x in r]
    db.close()
    return jsonify(dx)

@app.route('/search', methods=['POST'])
def search():
    if not request.json:
        return jsonify("error")
    print request.json
    keywords = request.json['field-keywords']
    limit = request.json['limit']
    # print category
    # print limit

    db = DBUtil()
    r = db.search(keywords, limit)
    dx = [{"asin" : x[0], 'category' : x[1], 'price' : float(x[2]), 'brand' : x[3], 'title': x[4], 'image_link': x[5]} for x in r]
    db.close()
    return jsonify(dx)

@app.route('/get_bpr_recommendation/<string:reviewer_id>/')
def get_bpr_recommendation(reviewer_id):
    dx = []
    if reviewer_id in bprmodel.keys():
      asin_list = bprmodel[reviewer_id]
      print asin_list[u'Recommendation list']
      db = DBUtil()
      r = db.get_product_item(asin_list[u'Recommendation list'])
      dx = [{"asin": x[0], 'title': x[1], 'image_link': x[2], 'price' : float(x[3])} for x in r]
      db.close()
    return jsonify(dx) 


@app.route('/get_bpr_product_categories/<string:reviewer_id>/')
def get_bpr_product_categories(reviewer_id):
    from random import randint

    d = {
        'single' : [],
        'multi'  : []
    }

    template = [3,3,1,3,4,1,1,1,1,1,1,1,1,4,4,1]

    if reviewer_id in bprmodel.keys():
      asin_list = bprmodel[reviewer_id]

      db = DBUtil()
      r = db.breakdown_categories(asin_list[u'Recommendation list'])

      for x in r:
          if x[1] > 3:
              d['multi'].append(x[0])
          else:
              d['single'].append(x[0])
      db.close()

    categories = []

    for x in template:
            if x == 1:                                                              
                categories.append( d['single'].pop(randint(0, len(d['single']) - 1)) )
            else:
                categories.append( d['multi'].pop(randint(0, len(d['multi']) - 1)) )

    dx = [{"category": x} for x in categories]                

    return jsonify(dx)     

@app.route('/get_reviewers/<int:top>/')
def get_reviewers(top):
    retval = {'result' : bprmodel.keys()[0:top]}
    return jsonify(retval)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
