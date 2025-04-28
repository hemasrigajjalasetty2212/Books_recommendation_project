import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import random

# Load data
users = pd.read_csv('Users.csv', encoding='ISO-8859-1')
books = pd.read_csv('Book.csv', encoding='ISO-8859-1')
ratings = pd.read_csv('Ratings.csv', encoding='ISO-8859-1')

# Data preprocessing
for i in users:
    users['Country']=users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')

#drop location column
users.drop('Location',axis=1,inplace=True)

users['Country']=users['Country'].astype('str')

users['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                           ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)

# outlier data became NaN
users.loc[(users.Age > 100) | (users.Age < 5), 'Age'] = np.nan

users['Age'] = users['Age'].fillna(users.groupby('Country')['Age'].transform('median'))

users['Age'].fillna(users.Age.mean(),inplace=True)


# books

books['Year-Of-Publication']=books['Year-Of-Publication'].astype('str')

#From above, it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','Year-Of-Publication'] = 2000
books.loc[books.ISBN == '0789466953','Book-Author'] = "James Buckley"
books.loc[books.ISBN == '0789466953','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','Book-Title'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','Year-Of-Publication'] = 2000
books.loc[books.ISBN == '078946697X','Book-Author'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','Publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','Book-Title'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

#rechecking
books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X'),:]
#corrections done

#making required corrections as above, keeping other fields intact
books.loc[books.ISBN == '2070426769','Year-Of-Publication'] = 2003
books.loc[books.ISBN == '2070426769','Book-Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','Publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','Book-Title'] = "Peuple du ciel, suivi de 'Les Bergers"

#books.loc[books.ISBN == '2070426769',:]

books['Year-Of-Publication']=pd.to_numeric(books['Year-Of-Publication'], errors='coerce')

books.loc[(books['Year-Of-Publication'] > 2006) | (books['Year-Of-Publication'] == 0),'Year-Of-Publication'] = np.NAN

#replacing NaNs with median value of Year-Of-Publication
books['Year-Of-Publication'].fillna(round(books['Year-Of-Publication'].median()), inplace=True)

#dropping last three columns containing image URLs which will not be required for analysis
books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'],axis=1,inplace=True)

#Filling Nan of Publisher with others
books.Publisher.fillna('other',inplace=True)

#Filling Nan of Book-Author with others
books['Book-Author'].fillna('other',inplace=True)


#ratings

ratings_new = ratings_new[ratings_new['User-ID'].isin(users['User-ID'])]


#Hence segragating implicit and explict ratings datasets
ratings_explicit = ratings_new[ratings_new['Book-Rating'] != 0]
ratings_implicit = ratings_new[ratings_new['Book-Rating'] == 0]

rating_count = pd.DataFrame(ratings_explicit.groupby('ISBN')['Book-Rating'].count())

most_rated_books = pd.DataFrame(['0316666343', '0971880107', '0385504209', '0312195516', '0060928336'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')

# Create column Rating average
ratings_explicit['Avg_Rating']=ratings_explicit.groupby('ISBN')['Book-Rating'].transform('mean')
# Create column Rating sum
ratings_explicit['Total_No_Of_Users_Rated']=ratings_explicit.groupby('ISBN')['Book-Rating'].transform('count')


# marging all dataset

Final_Dataset=users.copy()
Final_Dataset=pd.merge(Final_Dataset,ratings_explicit,on='User-ID')
Final_Dataset=pd.merge(Final_Dataset,books,on='ISBN')

# ...

# Create matrix
def create_matrix(Final_Dataset):
    matrix = Final_Dataset.pivot(index="User-ID", columns="book_id", values="Book-Rating")
    return matrix


# Build recommender
def build_recommender(matrix):
    nn = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
    nn.fit(csr_matrix(matrix))
    return nn

# Get recommendations
def get_recommendations(nn, matrix, user_id, book_id=None):
    distances, indices = nn.kneighbors(matrix.loc[user_id].values.reshape(1, -1))
    recs = []
    for i in indices[0]:
        recs.append(matrix.columns[i])
    return recs

# Streamlit app
def main():
    st.title("Book Recommendation System")

    # User input
    st.subheader("Enter User ID:")
    user_id = st.number_input("Enter User ID:", min_value=1, max_value=users.shape[0])

    # Book input
    st.subheader("Enter Book ID (optional):")
    book_id = st.number_input("Book ID (optional):", min_value=1, max_value=books.shape[0])

    # Get recommendations
    if st.button("Get Recommendations"):
        matrix = create_matrix(Final_Dataset)
        nn = build_recommender(matrix)
        recs = get_recommendations(nn, matrix, user_id-1, book_id-1) if book_id else get_recommendations(nn, matrix, user_id-1)
        st.write("Recommended Books:")
        for rec in recs:
            book_title = books.loc[books["book_id"] == rec, "title"].values[0]
            st.write(book_title)

if __name__ == "__main__":
    main()
