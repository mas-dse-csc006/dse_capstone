drop view if exists electronics_review_details_v cascade ;

create or replace view electronics_review_details_v as
        with rvd AS (
         SELECT 
            (j.data->>'reviewerID'::text) reviewerid,
            (j.data->>'reviewerName'::text) reviewername,
            COALESCE(p.price,0) AS price,
            p.brand AS brand,
            p.cleaned_brand,
            j.data ->> 'reviewText'::text AS reviewtext,
            j.data ->> 'unixReviewTime'::text AS unixtime,
    		j.data ->> 'summary'::text AS summary,
    		j.data ->> 'overall'::text AS overall,
    		to_date(j.data ->> 'reviewTime'::text, 'MM DD,YYYY'::text) AS reviewdate,
            --season_meteorological(ur.reviewdate) AS reviewdate_season,
            p.asin,
            p.categories,
            p.category_tags,
            p.also_viewed,
            p.also_bought,
            p.bought_together,
            p.image_url,
            p.title,
            p.title_fw,
            --r.product_hotcoded,
            --s.polarity,
            l4.level4_electronics level4,
            l4.avg_price level4_average,
            (p.price - l4.avg_price) l4_avg_adjusted
			from product_mv p
             inner join json_reviews_electronics_5 j 
            		on j.data->>'asin' = p.asin
             --don't have sentiment calculation for electronics
             --inner join in_calculated_sentiment s ON s.asin = p.asin
             inner join l4_electronics_summary_v l4 
            		on l4.level4_electronics = p.level4_electronics
        )
 SELECT 
    rvd.reviewerid,
    rvd.reviewername,
    rvd.price,
    rvd.level4,
    rvd.level4_average,
    rvd.l4_avg_adjusted,
    rvd.asin,
    rvd.title,
    rvd.title_fw,
    rvd.brand,
    rvd.cleaned_brand,
    rvd.reviewtext,
    rvd.summary,
    rvd.overall::numeric AS overall,
    rvd.unixtime,
    rvd.reviewdate,
    --rvd.reviewdate_season,
    rvd.categories,
    rvd.category_tags,
    rvd.also_viewed,
    rvd.also_bought,
    rvd.bought_together,
    rvd.image_url
    /*
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
      */
   FROM rvd
   where category_tags @> array['Computers & Accessories'];
   
--select * from electronics_review_details_v limit 20;

--select distinct tag from category_tag_v where electronics_flag = 'Y'
--order by tag;
