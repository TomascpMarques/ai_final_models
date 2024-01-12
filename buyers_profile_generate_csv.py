import csv
import random
import time


# Function to generate random Unix timestamp within a given range
def generate_unix_timestamp(start, end):
    return str(random.randint(start, end))


# Function to generate random product category
def generate_product_category():
    categories = ["Consumer Electronics", "Beauty", "Home Care", "Furniture", "Phones", "Informatics", "Printers",
                  "Screens", "Televisions", "Audio"]
    return random.choice(categories)


# Function to generate random product discount
def generate_product_discount():
    return round(random.uniform(0, 0.8), 2)


# Function to generate random entry for "SEEN PRODUCTS"
def generate_seen_product_entry(user_id):
    category = generate_product_category()
    discount = generate_product_discount()
    bought = generate_unix_timestamp(1609459200, 1640995200) + str(
        random.randint(0, 4))  # Example range: Jan 1, 2023, to Jan 1, 2024
    return [category, discount, bought, str(user_id)]


# Generate and write "SEEN PRODUCTS" data to CSV
with open('seen_products_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Product Category", "Product Discount", "Product Bought", "User ID"])

    for _ in range(2000):  # Adjust the number of entries as needed
        csv_writer.writerow(generate_seen_product_entry(random.randint(0, 900)))


# Function to generate random entry for "BOUGHT PRODUCTS"
def generate_bought_product_entry(user_id):
    date_buy = generate_unix_timestamp(1609459200, 1640995200)  # range Jan 1, 2023, to Jan 1, 2024
    op = random.uniform(15, 120)
    dc = random.uniform(0, 0.8)
    bp = op - (op * dc)
    buy_price = "{:.1f}".format(bp)  # $10 to $100
    original_price = "{:.1f}".format(op)  # $15 to $120
    discount = "{:.1f}".format(dc * 100)
    release_date = generate_unix_timestamp(1609459200, 1640995200)
    category = generate_product_category()
    season_bought = random.randint(0, 4)  # Example seasons: 1 to 5
    return [date_buy, discount, buy_price, original_price, release_date, category, season_bought, str(user_id)]


# Generate and write "BOUGHT PRODUCTS" data to CSV
with open('bought_products_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(
        ["Date of Product Buy", "Discount", "Product Buy Price", "Product Original Price", "Product Release Date",
         "Product Category", "Season Bought", "User ID"])

    for _ in range(2000):  # Adjust the number of entries as needed
        csv_writer.writerow(generate_bought_product_entry(random.randint(0, 900)))


# Function to generate random entry for "USER CLASSIFIERS"
def generate_user_classifier_entry(user_id):
    gender = random.choice(["male", "female"])
    age = random.randint(18, 80)
    session_time = random.randint(60, 3600)  # seconds
    active_seasons = random.sample(["Spring", "Summer", "Winter", "Fall", "Holidays"], 3)
    return [gender, age, session_time, active_seasons, str(user_id)]


# Generate and write "USER CLASSIFIERS" data to CSV
with open('user_classifiers_data.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Gender", "Age", "Average Session Time", "Most Active Seasons", "User ID"])

    for user_id in range(901):  # User IDs from 0 to 900
        csv_writer.writerow(generate_user_classifier_entry(user_id))
