"""

/*
Enter your query below.
Please append a semicolon ";" at the end of the query
*/


select
  b.buyer_id BUYER_ID,
  b.total_worth TOTAL_WORTH
from
  (
    select
      a.buyer_id buyer_id,
      COUNT(a.house_id) house_count,
      SUM(a.price) total_worth
    from
      (
        select
          house.BUYER_ID buyer_id,
          house.HOUSE_ID house_id,
          price.PRICE price
        from
          house
          left join price on house.HOUSE_ID = price.HOUSE_ID
      ) as a
    group by
      a.buyer_id
  ) as b
where
  b.house_count > 1
  and b.total_worth >= 100;


"""