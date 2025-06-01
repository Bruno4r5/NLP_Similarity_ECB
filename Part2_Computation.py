import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

path = r"C:\Users\JABAMI\Desktop\Cours_Sorbonne_FTD\applied_data_science\rendu_final"
df = pd.read_excel(path + "\database.xlsx", dtype=str)
df = df.drop(df.columns[0], axis=1)
df.columns = ["link", "title", "date", "content"]  

as_paper = False # To find the same results as the paper, set it to True
plot_ok = False # To find the same results as the paper, set it to True



# We convert the date
def convert_date(date):
    year = str(date)[:2]
    if int(year) <= 24:  # if the 2 first digits are less than 24 for 2024
        return "20" + str(date)
    else:  # if the 2 first digits are more than 24 for the late 90
        return "19" + str(date)

#df["date"] = pd.to_datetime(df["date"].apply(convert_date), format='%Y%m%d', errors='coerce').dt.strftime("%Y-%m-%d")
df.set_index("date", inplace=True)

####################################
########### DATA CLEANING ##########
####################################

df = df[df.index >= "1999-01-01"] # As the paper

rows_to_keep = ["ECB Press conference","ECB Press Conference","Introductory statement","Transcript of the Press Briefing","PRESS CONFERENCE"] # The only rows that we want to keep
df = df[df["title"].str.contains('|'.join(rows_to_keep), case=False)]
rows_to_drop = ["on the winning design", "joint"] # In the rows we have ketpt, there is still wrong rows that we need to remove
df = df[~df["title"].str.contains('|'.join(rows_to_drop), case=False)]

# verify if we don't have two meetings the same day
duplicate_indices = df.index[df.index.duplicated(keep=False)]
duplicates = df.loc[duplicate_indices]

# having a look at the contents occuring the same month
df.index = pd.to_datetime(df.index)
month_year = df.index.to_period('M')  
duplicates_my = df[month_year.duplicated(keep=False)]
# Thanks to it, investigating and finding extract of meetings that are not MRO announcments
df = df.drop(pd.to_datetime(["2001-12-13", "2003-09-17", "2005-01-20"]))
df = df.drop(pd.to_datetime(["2003-10-13", "2014-10-26"]))

# As the first sentence don't bring any meaning and is always composed of positive words such as "good" in "good afternoon", we remove it
def remove_first_sentence(text):
    sentences = text.split('.', 1) 
    return sentences[1].strip() if len(sentences) > 1 else text
df["content"] = df["content"].apply(remove_first_sentence)

df["content_length_before"] = df["content"].apply(len)

# Now we drop the Q&A part of the content
df["content"] = df["content"].str.split("We are now ready to take your questions").str[0]
df["content"] = df["content"].str.split("We are now at your disposal").str[0]
df["content"] = df["content"].str.split("I am now at your disposal for questions").str[0]
df["content"] = df["content"].str.split("Transcript of the questions asked and the answers given by").str[0]
df["content"] = df["content"].str.split("We stand ready to answer any questions you may have").str[0]
df["content"] = df["content"].str.split("My first question would be about").str[0]
df["content"] = df["content"].str.split("Question:").str[0]

df["content_length_after"] = df["content"].apply(len)

# just a last verification
QandA_not_removed = df.apply(lambda row: row["content"] if row["content_length_before"] == row["content_length_after"] else None, axis=1)
QandA_not_removed = QandA_not_removed.dropna()
df = df.drop(columns=["content_length_before", "content_length_after"])

####################################
########### TOKENIZATION ###########
####################################

import nltk # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

my_stopwords = stopwords.words("english")
stemmer = PorterStemmer()

df["tokens"] = df["content"].apply(lambda speech: [
    stemmer.stem(token)
    for token in word_tokenize(re.sub("[^a-zA-Z]", " ", speech.lower()))
    if token not in my_stopwords])


####################################
########### PESSIMISM    ###########
####################################

df['pessimism'] = np.NaN

# We load the dictionnary used in the paper 
sentiment_words = pd.read_csv(path + "\Loughran-McDonald_MasterDictionary_1993-2023.csv")


# We extract the positive and negative words in the dictionnary. Removing the updates after 2013 for the "replication of the paper" setup 
if as_paper == True:
    neg_words = set(sentiment_words[(sentiment_words["Negative"] != 0) & (sentiment_words["Negative"] < 2014)]["Word"].str.lower())
    pos_words = set(sentiment_words[(sentiment_words["Positive"] != 0) & (sentiment_words["Negative"] < 2014)]["Word"].str.lower())
else: 
    neg_words = set(sentiment_words[sentiment_words["Negative"] != 0]["Word"].str.lower())
    pos_words = set(sentiment_words[sentiment_words["Positive"] != 0]["Word"].str.lower())


# Function to calculate the pessimism score
def calculate_pessimism(row):
    neg_tokens = [token for token in row['tokens'] if token in neg_words]
    pos_tokens = [token for token in row['tokens'] if token in pos_words]
    
    neg_count = len(neg_tokens)
    pos_count = len(pos_tokens)
    total_words = len(row['tokens'])
    
    if total_words == 0:
        return np.NaN, np.NaN
    
    pessimism_score = ((neg_count - pos_count) / total_words)*100 # --> Percentage
    return pessimism_score


# We apply the function to the DataFrame
df['pessimism'] = df.apply(calculate_pessimism, axis=1)

if as_paper == True:
    print(df[(df.index.year <= 2013)]["pessimism"].mean())
else:
    print(df["pessimism"].mean())


####################################
########### SIMILARITY   ###########
####################################

from itertools import tee
import numpy as np

df=df.sort_values(by='date')

def jaccard_similarity(set_1, set_2):
    intersection = set_1.intersection(set_2)
    union = set_1.union(set_2)
    return len(intersection) / len(union) if len(union) > 0 else 0

def find_bigrams(input_list):
    a, b = tee(input_list)
    next(b, None)
    return set(zip(a, b))

jaccard_similarities = [np.nan] 

# We iterate over the DataFrame and calculate Jaccard similarity for bigrams of successive speeches
for i in range(1, len(df['tokens'])):
    bigrams_i = find_bigrams(df['tokens'].iloc[i])
    bigrams_i_1 = find_bigrams(df['tokens'].iloc[i - 1])
    similarity = jaccard_similarity(bigrams_i, bigrams_i_1)
    jaccard_similarities.append(similarity)

df['jaccard_similarity'] = jaccard_similarities

if plot_ok == True:
    # To get the same look as the one in the paper
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['jaccard_similarity'], color='gray', linewidth=1)

    plt.title("Jaccard Similarity Over Time ", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Similarity", fontsize=12)
    plt.ylim(0, 0.7)  
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)  
    plt.tight_layout()

    plt.show()


####################################
########### ADD DATA     ###########
####################################

########### MRO DATA

mro = pd.read_csv(path + "\mro_data.csv")
df = df.reset_index()

mro.columns = ["date", "date_to_del", "mro"]
mro = mro.drop(columns = "date_to_del")
mro["date"] = pd.to_datetime(mro["date"])
mro["mro_change"] = mro["mro"] - mro["mro"].shift(1)

# filter for mro_changes different from zero
mro_changes = mro[mro["mro_change"] != 0][["date", "mro_change"]]

# add a column in df for the results
df["date"] = pd.to_datetime(df["date"])
df["mro_change"] = 0  # initialize to 0

# associate the dates with the nearest date in df
for _, row in mro_changes.iterrows():
    closest_date_index = (df["date"] - row["date"]).abs().idxmin()
    df.loc[closest_date_index, "mro_change"] = row["mro_change"]

df.set_index("date", inplace=True)


########### GDP DATA TO GET THE OUTPUT GAP

import statsmodels.api as sm

gdp_data = pd.read_csv(path + "\gdp_data.csv")
#gdp_data['date'] = pd.to_datetime(gdp_data['date'])
gdp_data['date'] = pd.to_datetime(gdp_data['date'], format='%d/%m/%Y', dayfirst=True)
gdp_data.set_index('date', inplace=True)
gdp_data = gdp_data[gdp_data.index >= "1999-01-01"]

# We apply the HP Filter to calculate potential GDP and output gap
if as_paper == True:
    actual_gdp = gdp_data[gdp_data.index <= '2013-12-31']['gdp']
else:
    actual_gdp = gdp_data['gdp']
cycle, trend = sm.tsa.filters.hpfilter(actual_gdp, lamb=1600)

gdp_data["potential_GDP"] = trend
gdp_data["substraction"] = cycle
gdp_data['output_gap'] = (gdp_data['substraction'] / gdp_data['potential_GDP'])*100 # Output gap en pourcentage 


# We Align GDP data (which are quarterly) to the nearest announcement date
df = pd.merge_asof(df, gdp_data, on='date', direction='nearest')
df = df.drop(columns=["substraction"])
df.set_index('date', inplace=True)

print(gdp_data["output_gap"].describe())


########### HCPI DATA

hcpi_monthly = pd.read_csv(path + "\hcpi_data.csv")
hcpi_monthly['date'] = pd.to_datetime(hcpi_monthly['date'], format='%d/%m/%Y', dayfirst=True)
hcpi_monthly.set_index('date', inplace=True)
hcpi_monthly = hcpi_monthly[hcpi_monthly.index >= "1999-01-01"]

df = pd.merge_asof(df, hcpi_monthly, on='date', direction='nearest')
df.set_index('date', inplace=True)

df['hcpi_change'] = df['hcpi'].pct_change() * 100


########### ADD LOG SIMILARITY / LOG TIME / TIME COUNT / PESSIMISM * SIMILARITY

# Calculate 'Time' (days since the first announcement date)
#df['Time'] = (df.index - df.index.min()).days 
df['Time'] = (df.index - pd.to_datetime("1999-01-01")).days # 1999/01/01 the begining of the sample period (and we do as did in the paper)

df['logSimilarity'] = np.log(df['jaccard_similarity'])
df['logTime'] = np.log(df['Time'])
df['time_count'] = range(1, len(df) + 1)
df['pess_x_sim'] = df['pessimism'] * df['logSimilarity']

# SX5E historical price data
sx5e_hist_price = pd.read_csv(path + "/SX5E_hist_price.csv")

sx5e_hist_price['Date'] = pd.to_datetime(sx5e_hist_price['Date'], format='mixed')

sx5e_hist_price.set_index('Date', inplace=True)
sx5e_hist_price = sx5e_hist_price.sort_index()

# We calculate the mean of the daily return over a period of 200 days between 50 days and 250 days before the date
sx5e_hist_price['mean_Daily_Returns_50_250'] = sx5e_hist_price['Daily Returns'].rolling(window=200, min_periods=200).mean().shift(50)

# We compute the difference between the daily return and the mean_Daily_Returns_50_250
sx5e_hist_price['abnormal_returns'] = sx5e_hist_price['Daily Returns'] - sx5e_hist_price['mean_Daily_Returns_50_250']

# We calculate the rolling sum of the differences in daily returns between -5 and +5 days
sx5e_hist_price['CAR'] = sx5e_hist_price['abnormal_returns'].rolling(window=11, center=True).sum()

sx5e_hist_price['abs_CAR'] = sx5e_hist_price['CAR'].abs()

# We merge 
df = pd.merge_asof(df, sx5e_hist_price[['mean_Daily_Returns_50_250', 'abnormal_returns','CAR', 'abs_CAR']], left_on='date', right_index=True, direction='nearest')



if plot_ok == True:
    ################# DESCRIPTIVE STATISTICS

    columns = ["CAR", "abs_CAR", 'pessimism', 'jaccard_similarity', 'output_gap',  'hcpi_change', 'mro_change']

    # Step 1: Compute descriptive statistics
    stats = df[columns].describe(percentiles=[0.25, 0.5, 0.75]).T  # Transpose for a cleaner layout

    # Step 2: Select only the required statistics and rename for clarity
    stats = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats.columns = ['Mean', 'Std. Dev.', 'Min.', 'Quartile 1', 'Median', 'Quartile 3', 'Max.']

    stats = stats.round(4)

    # Step 3: Create a table and save it as a PDF
    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=stats.values, colLabels=stats.columns, rowLabels=stats.index, cellLoc='center', loc='center')

    # Save the table as a PDF
    pdf_path = path + "\stats_table.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    print(f"Table saved as {pdf_path}")


    ###################### CREATE FIGURE 1 IN SECTION 2

    df_modif = df.dropna(subset=["mro_change"])

    df_modif["year"] = df_modif.index.year

    bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 1]
    labels = ["-0.75", "-0.5", "-0.25", "0", "0.25", "0.5", "0.75"]

    # categorize mro_change into bins
    df_modif["mro_change_bin"] = pd.cut(df_modif["mro_change"], bins=bins, labels=labels, right=True)

    # create the crosstab
    table = pd.crosstab(df_modif["year"], df_modif["mro_change_bin"], margins=True, margins_name="Total")
    print(table)

    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(table, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.title("MRO Change Distribution by Year", fontsize=16)
    plt.xlabel("MRO Change Bin", fontsize=14)
    plt.ylabel("Year", fontsize=14)
    plt.tight_layout()
    plt.show()



########################################################################
########### The 4 OLS REGRESSION explaining logSimilarity ##############
########################################################################


df['mro_change'] = pd.to_numeric(df['mro_change'], errors='coerce')
df_regSimilarity = df.drop(index=pd.to_datetime("1999-01-07"))
y = df_regSimilarity['logSimilarity']

# Define the 4 regression models
regressions = {
    "Intercept, Output Gap, Inflation, MRO Change": ['output_gap', 'hcpi_change', 'mro_change'],
    "Intercept and Time": ['logTime'],
    "Intercept, Time, Output Gap, Inflation, MRO Change": ['logTime', 'output_gap', 'hcpi_change', 'mro_change'],
    "Intercept, Time (count), Output Gap, Inflation, MRO Change": ['time_count', 'output_gap', 'hcpi_change', 'mro_change'],
}

results = {}

for name, variables in regressions.items():
    X = df_regSimilarity[variables]
    X = sm.add_constant(X)  # the intercept
    model = sm.OLS(y, X).fit()
    
    # To get the coefficients and p-values
    coefficients = model.params
    p_values = model.pvalues

    # We add significance stars based on p-values
    coefficients_with_significance = coefficients.apply(lambda coef: 
        f"{coef:.4f}" + 
        ('***' if p_values[coefficients.index[coefficients == coef][0]] < 0.01 else 
        '**' if p_values[coefficients.index[coefficients == coef][0]] < 0.05 else 
        '*' if p_values[coefficients.index[coefficients == coef][0]] < 0.10 else ''))
    
    results[name] = {
        'Intercept': coefficients_with_significance,
        'Adjusted R-squared': model.rsquared_adj
    }

# Diplaying
for name, result in results.items():
    print(f"Regression: {name}")
    print("Coefficients:")
    print(result['Intercept'])
    print(f"Adjusted R-squared: {result['Adjusted R-squared']:.4f}")
    print("-" * 50)




########################################################################
############## The 3 OLS REGRESSION explaining abs(CAR) ################
########################################################################

#df_regAbsCAR = df.dropna()

df_regAbsCAR = df[df.index > pd.to_datetime("1999-12-02")]
y = df_regAbsCAR['abs_CAR']

# Define the 3 regression models
regressions = {
    "Intercept, Pessimism and R²": ['pessimism'],
    "Intercept, Output Gap, Inflation, MRO Change and R²": ['output_gap', 'hcpi_change', 'mro_change'],
    "Intercept, Pessimism * Similarity and R²": ['pess_x_sim'],
    "Intercept, Pessimism * Similarity, Output Gap, Inflation, MRO Change and R²": ['pess_x_sim', 'output_gap', 'hcpi_change', 'mro_change']
}

results = {}

for name, variables in regressions.items():
    X = df_regAbsCAR[variables]
    X = sm.add_constant(X)  # The intercept
    model = sm.OLS(y, X).fit()
    
    # To get the coefficients and p-values
    coefficients = model.params
    p_values = model.pvalues

    # We add significance stars based on p-values
    coefficients_with_significance = coefficients.apply(lambda coef: 
        f"{coef:.4f}" + 
        ('***' if p_values[coefficients.index[coefficients == coef][0]] < 0.01 else 
        '**' if p_values[coefficients.index[coefficients == coef][0]] < 0.05 else 
        '*' if p_values[coefficients.index[coefficients == coef][0]] < 0.10 else ''))

    results[name] = {
        'Intercept': coefficients_with_significance,
        'Adjusted R-squared': model.rsquared_adj
    }

# Diplaying
for name, result in results.items():
    print(f"Regression: {name}")
    print("Coefficients:")
    print(result['Intercept'])
    print(f"Adjusted R-squared: {result['Adjusted R-squared']:.4f}")
    print("-" * 50)




###################################################################################################
############## 2 OLS REGRESSION explaining abs(CAR) using datas from our extension ################
###################################################################################################

df_extension = pd.read_csv(path + "\extension.csv")
df_extension.set_index("date", inplace=True)
df_extension.index = pd.to_datetime(df_extension.index)
df_extension = df_extension[df_extension.index > pd.to_datetime("1999-12-02")]

df_regAbsCAR["Subjectivity"] = df_extension["TextBlob_Subjectivity"]


df_regAbsCAR = df_regAbsCAR.dropna()
y = df_regAbsCAR['abs_CAR']

# Define the 2 regression models
regressions = {
    "Intercept, Subjectivity, Output Gap, Inflation, MRO Change and R²": ['Subjectivity','output_gap', 'hcpi_change', 'mro_change'],
    "Intercept, Subjectivity, Pessimism, Pessimism * Similarity  and R²": ['Subjectivity', 'pessimism', 'pess_x_sim']
}

results = {}

# Run regressions and store results
for name, variables in regressions.items():
    X = df_regAbsCAR[variables]
    X = sm.add_constant(X)  # The intercept
    model = sm.OLS(y, X).fit()
    
    # To get the coefficients and p-values
    coefficients = model.params
    p_values = model.pvalues

    # We add significance stars based on p-values
    coefficients_with_significance = coefficients.apply(lambda coef: 
        f"{coef:.4f}" + 
        ('***' if p_values[coefficients.index[coefficients == coef][0]] < 0.01 else 
        '**' if p_values[coefficients.index[coefficients == coef][0]] < 0.05 else 
        '*' if p_values[coefficients.index[coefficients == coef][0]] < 0.10 else ''))

    results[name] = {
        'Intercept': coefficients_with_significance,
        'Adjusted R-squared': model.rsquared_adj
    }

# Displaying
for name, result in results.items():
    print(f"Regression: {name}")
    print("Coefficients:")
    print(result['Intercept'])
    print(f"Adjusted R-squared: {result['Adjusted R-squared']:.4f}")
    print("-" * 50)


