show databases;

/*CREATE DATABASE*/
CREATE DATABASE movierecom;
SHOW DATABASES;
USE movierecom;

/*CREATE PREFERANCE TABLE*/
CREATE TABLE user_preference(
id int(5),
user_age INT,
preferred_genres varchar(255),
age_rating varchar(100),
created TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

/*CREATE WATCH HISTORY TABLE*/
CREATE TABLE watch_history(
id int(5),
movie_title varchar(50),
age_rating varchar(100),
watch_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

/* SEE THE TABLE */
SELECT *
FROM watch_history;