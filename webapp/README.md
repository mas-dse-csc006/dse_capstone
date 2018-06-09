# webapp
Web app for amazon recommender system

## Architecture

web -> nginx reverse proxy -> flask app server <-> postgres/redis DB

I image this acting like an REST API where you can get rankings like:

```
GET /users/[userid]/rankings?limit=10
```

which would return the top ten items in json:

```
{
  [
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r},
    {asin: "xxxxxx", image_url:"http://...jpg", rank: r.r}
  ]
}
```

## Setup

```
docker-compose build
docker-compose up
```

Then it should be running on port 3000

## Misc

Shell:

```
docker-compose run --entrypoint /bin/ash web
```


Query to get users w/ most reviews

```
select reviews.aid, users.id, count(*) cnt
from reviews
join users on users.aid = reviews.aid
group by reviews.aid, users.id
order by cnt desc
```

## K8s shell

kubectl exec -ti $POD --namespace sharknado-recsys -- bash


Export local DB
pg_dump -U postgres -h localhost -c sharknado-web  > database.sql
 
To k8s remote:

cat database.sql | kubectl exec -i $POD --namespace sharknado-recsys -- psql -U postgres -d sharknado-web