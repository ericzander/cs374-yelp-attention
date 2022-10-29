# Yelp Dataset Overview

### Source

https://www.yelp.com/dataset

### Description

Yelp provides a large dataset consisting of natural language, images,
and other metrics for describing businesses.

The subset of the offered data to be used in this project consists of
user-generated reviews of businesses. These are found in 
*yelp_academic_dataset_review.json*, though in a format easily read as a
Pandas DataFrame. The features of this file are below.

6,990,280 user reviews are included.

### Features

* review_id (str): Unique identifier for review
* user_id (str): User identifier
* business_id (str): Business identifier
* stars (int): 5-star review rating, 1-5
* useful (int): Number of users who marked review as useful
* funny (int): Number of users who marked review as funny
* cool (int): Number of users who marked review as cool
* text (str): Content of review 
* date (str): Date and time of review, "YYYY-MM-DD HH:MM:SS"

Only the *text* and *star* features will be used for this project. They will
serve as input and labels respectively.

### Examples

* 'I thoroughly enjoyed the show.  Chill way to spend a Friday night.'
* 'Rude and unprofessional staff. I called today to find out if they had availability over the Thanksgiving holiday and the staff was short and basically said no then hung up the phone. I will take my business elsewhere.'
* 'Very cute place and good food. Pear cider beer was really good the service was really good and great ambiance and the price was very reasonable'
* 'Best WalMart on the southside. Nice location. If you're looking for a southside WalMart, this is the one to visit.\n\nProduce tends to be good and fresh because they move a high volume. The produce selection is not great for unusual stuff, but the basics are fresh.'
* 'What an insanely friendly staff! The selection was amazing. Such beautiful dresses and accessories. We were incredibly pleased with our visit here. Also they had several chicken-themed accessories. What a lovely visit. Merçi beaucoup á la France!'