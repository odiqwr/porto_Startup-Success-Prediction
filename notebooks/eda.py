# Load Dataset
df = pd.read_csv('startup data.csv')
df.head(10)

# Checking Structure Data
df.info()
numeric=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num=df.select_dtypes(include=numeric)
df_cat=df.select_dtypes(include='object')

# Analysis Descriptive Statistics
describeNum = df.describe(include =['float64', 'int64', 'float', 'int'])      # for numeric analysis
describeNum.T.style.background_gradient(cmap='viridis',low=0.2,high=0.1)      # for show analysis results
describeNumCat = df.describe(include=["O"])                                   # for object analysis
describeNumCat.T.style.background_gradient(cmap='viridis',low=0.2,high=0.1)

# Checking Missing Values
null=pd.DataFrame(df.isnull().sum(),columns=["Null Values"])
null["% Missing Values"]=(df.isna().sum()/len(df)*100)
null = null[null["% Missing Values"] > 0]
null.style.background_gradient(cmap='viridis',low =0.2,high=0.1) 

# Handling Missing Values
df['Unnamed: 6'] = df.apply(lambda row: (row.city) + " " + (row.state_code) + " " +(row.zip_code)  , axis = 1)      # to handle missing values ​​in column Unnamed: 6
df['closed_at'] = df['closed_at'].fillna(value="31/12/2013")                                                        # to handle missing values ​​in column closed_at, with fixed date assumption
df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(value="0")                                   # to handle missing values ​​in column age_first_milestone_year, with assumption no milestone
df['age_last_milestone_year'] = df['age_last_milestone_year'].fillna(value="0")                                     # to handle missing values ​​in column age_last_milestone_year, with assumption no milestone
df.drop(["state_code.1"], axis=1, inplace=True)                                                                     # to handle missing values ​​in column state_code.1, because values in column state_code.1 = state_code

# Graphic Approach
# heatmap correlation for all features
df.corr() 
features = ['age_first_funding_year','age_last_funding_year','age_first_milestone_year','age_last_milestone_year','relationships','funding_rounds','funding_total_usd','milestones','is_CA','is_NY','is_MA','is_TX','is_otherstate','is_software','is_web','is_mobile','is_enterprise','is_advertising','is_gamesvideo','is_ecommerce','is_biotech','is_consulting','is_othercategory','has_VC','has_angel','has_roundA','has_roundB','has_roundC','has_roundD','avg_participants','is_top500','status']
plt.figure(figsize=(30,20))
ax = sns.heatmap(data = df[features].corr(),cmap='YlGnBu',annot=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5,top - 0.5)     

# heatmap correlation for top 10 features against status
cols = df[features].corr().nlargest(10,'status')['status'].index
cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, cmap='YlGnBu', fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()                               

# scatter plot for correlation between two features
fig, ax = plt.subplots()
_ = plt.scatter(x=df['age_first_funding_year'], y=df['age_last_funding_year'], edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="age_first_funding_year", ylabel="age_last_funding_year")]

# box plots for outlier detection
featuresNum = ['age_first_funding_year','age_last_funding_year','age_first_milestone_year','age_last_milestone_year','relationships','funding_rounds','funding_total_usd','milestones','avg_participants']
plt.figure(figsize=(15, 7))
for i in range(0, len(featuresNum)):
    plt.subplot(1, len(featuresNum), i+1)
    sns.boxplot(y=df[featuresNum[i]], color='green', orient='v')
    plt.tight_layout()

# pie plot for proportion status startup analysis
df_acquired = df[(df["status"] == True)]
df_closed = df[(df["status"] == False)]
value_counts = df["status"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie(x=[value_counts[False], value_counts[True]], labels=['No', 'Yes'], colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
_ = ax.axis('equal')
_ = ax.set_title('Startup Acquired')

# bar plot to determine potential successful startup categories
fig, ax = plt.subplots(figsize=(10,7))
_ = sns.barplot(x="category_code", y="success_rate", data=most_success_rate, palette="nipy_spectral", ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
_ = ax.set(xlabel="Category", ylabel="Success Rate of Start Up")

