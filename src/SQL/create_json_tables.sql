drop table if exists json_product;
create table json_product (
    id serial not null primary key,
	data jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP 
) ;

drop table if exists json_reviews_clothing_Shoes_and_jewelry_5 ;
create table json_reviews_clothing_Shoes_and_jewelry_5 (
    id serial not null primary key,
	data jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP 
) ;

drop table if exists json_reviews_electronics_5 ;
create table json_reviews_electronics_5 (
    id serial not null primary key,
	data jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP 
) ;