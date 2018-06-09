drop view if exists review_details_v cascade ;

create or replace view review_details_v as
        with rvd AS (
         SELECT r.reviewerid,
            COALESCE(r.price,0) AS price,
            COALESCE(r.brand, p.brand::character varying) AS brand,
            ur.helpful,
            ur.reviewtext,
            ur.summary,
            ur.overall,
            ur.unixtime,
            ur.reviewdate,
            season_meteorological(ur.reviewdate) AS reviewdate_season,
            p.asin,
            p.categories,
            replace(p.category_tags,'''','')::text[] category_tags,
            p.also_viewed,
            p.also_bought,
            p.bought_together,
            p.image_url,
            p.title,
            r.product_hotcoded,
            s.polarity,
            /* budget: if use has purchase items in this level before use that average info as their
             * otherwise use the average price of the item in that level 4
             */
            l4.level4,
            bug.avg_price level4_budget,
            l4.avg_price level4_average,
            case 
            	when 
                	bug.l4_avg_price is not null and p.price != bug.avg_price 
                then p.price - bug.avg_price
                else p.price - l4.avg_price 
            end person_l4_budget,
            (p.price - l4.avg_price) l4_avg_adjusted
           FROM reviews_women_scraped_mv r
             inner JOIN unfold_reviews_clothing_5core ur on ur.asin = r.asin and ur.reviewerid = r.reviewerid
             inner join product_mv p ON p.asin = r.asin
             inner join in_calculated_sentiment s ON s.asin = p.asin
             inner join subcategory_l4_summary l4 on l4.level4 = p.level4
             left outer join reviewer_l4_summary bug on bug.reviewerid = r.reviewerid and p.level4 = bug.level4
        )
 SELECT rvd.reviewerid,
    rvd.price,
    rvd.level4,
    rvd.level4_budget,
    rvd.level4_average,
    rvd.person_l4_budget,
    rvd.l4_avg_adjusted,
    rvd.asin,
    rvd.title,
    rvd.brand,
    rvd.helpful,
    rvd.reviewtext,
    rvd.summary,
    rvd.overall::numeric AS overall,
    rvd.unixtime,
    rvd.reviewdate,
    rvd.reviewdate_season,
    rvd.categories,
    rvd.category_tags,
    rvd.also_viewed,
    rvd.also_bought,
    rvd.bought_together,
    rvd.image_url,
    rvd.polarity,
    case when rvd.polarity < -0.5 then 1 else 0 end very_negative,
	case when rvd.polarity >= -0.5 and polarity < -0.1 then 1 else 0 end negative,
	case when rvd.polarity >= -0.1 and polarity < 0.1 then 1 else 0 end as neutral, 
	case when rvd.polarity >= 0.1 and polarity < 0.5 then 1 else 0 end positive,
	case when rvd.polarity >= 0.5 then 1 else 0 end very_positive,
    rvd.product_hotcoded,
        CASE
            WHEN rvd.reviewdate_season = 'spring'::text THEN 1
            ELSE 0
        END AS spring_flag,
        CASE
            WHEN rvd.reviewdate_season = 'summer'::text THEN 1
            ELSE 0
        END AS summer_flag,
        CASE
            WHEN rvd.reviewdate_season = 'fall'::text THEN 1
            ELSE 0
        END AS fall_flag,
        CASE
            WHEN rvd.reviewdate_season = 'winter'::text THEN 1
            ELSE 0
        END AS winter_flag
   FROM rvd;
-- inner join 
-- 181,329
/*
select * from review_details_v
where asin = 'B000MX3SH2'
limit 10 ;

select count(distinct asin) from in_calculated_sentiment ;
--23,033

select count(reviewerid), count(distinct reviewerid), 
count(distinct asin) from reviews_women_scraped_mv;
--'181329','35358','14780'
-- After removal or duplicates, items without prices, suspect low and high prices
--'132006','34111','10321'
*/

--select count(reviewerid), count(distinct reviewerid), count(distinct asin) from review_details_v 
-- # records, # reviewers, # products
-- '182763','35358','14771'
-- After removal or duplicates, items without prices, suspect low and high prices
-- '131979','34107','10318'
/*
9 asins have odd categories. Clothing & Jewerly in one group; Women's in another;
 not; not comfortable changing right now; might try later and cross-check

select count(r.reviewerid), count(distinct r.reviewerid), count(distinct r.asin)
FROM reviews_women_scraped_cpp_fv r
             left outer JOIN unfold_reviews_clothing_5core ur on ur.asin = r.asin and ur.reviewerid = r.reviewerid
             inner join product_mv p ON p.asin = r.asin
             left outer join in_calculated_sentiment s ON s.asin::text = p.asin;
*/

--select count(reviewerid) from reviews_women_scraped_mv;
