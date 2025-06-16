# Handling Missing Values (because it has missing column values)
df['Unnamed: 6'] = df.apply(lambda row: (row.city) + " " + (row.state_code) + " " +(row.zip_code)  , axis = 1)      # to handle missing values ​​in column Unnamed: 6
df['closed_at'] = df['closed_at'].fillna(value="31/12/2013")                                                        # to handle missing values ​​in column closed_at, with fixed date assumption
df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(value="0")                                   # to handle missing values ​​in column age_first_milestone_year, with assumption no milestone
df['age_last_milestone_year'] = df['age_last_milestone_year'].fillna(value="0")                                     # to handle missing values ​​in column age_last_milestone_year, with assumption no milestone
df.drop(["state_code.1"], axis=1, inplace=True)                                                                     # to handle missing values ​​in column state_code.1, because values in column state_code.1 = state_code

# Checking Data Duplication
duplicate = df[df.duplicated()] 
print("Duplicate Rows :")

# Handling Negative Values (if it has negative values, drop with consent)
df=df.drop(df[df.age_first_funding_year<0].index)
df=df.drop(df[df.age_last_funding_year<0].index)
df=df.drop(df[df.age_first_milestone_year<0].index)
df=df.drop(df[df.age_last_milestone_year<0].index)

# Handling Outlier with Log Function
df["age_first_funding_year"] = np.log1p(df["age_first_funding_year"])
df["age_last_funding_year"] = np.log1p(df["age_last_funding_year"])
df["age_first_milestone_year"] = np.log1p(df["age_first_milestone_year"])
df["age_last_milestone_year"] = np.log1p(df["age_last_milestone_year"])
df["funding_total_usd"] = np.log1p(df["funding_total_usd"])

# Feature Engineering
df['has_RoundABCD'] = np.where((df['has_roundA'] == 1) | (df['has_roundB'] == 1) | (df['has_roundC'] == 1) | (df['has_roundD'] == 1), 1, 0)      # create new column has_RoundABCD with assuming that the company has funding in one of rounds a, b, c, d
df['has_Investor'] = np.where((df['has_VC'] == 1) | (df['has_angel'] == 1), 1, 0)                                                                # create new column has_Investor with assuming that the company has VC or angel
df['has_Seed'] = np.where((df['has_RoundABCD'] == 0) & (df['has_Investor'] == 1), 1, 0)                                                          # create new column has_Seed with assuming that the company has_Investor and has_RoundABCD
df['invalid_startup'] = np.where((df['has_RoundABCD'] == 0) & (df['has_VC'] == 0) & (df['has_angel'] == 0), 1, 0)                                # create new column startup invalid assuming that company has no_Investors and has no_RoundABCD
# convert year to datetime (days)
df.founded_at=pd.to_datetime(df.founded_at)
df.closed_at=pd.to_datetime(df.closed_at)
# calculate the time from standing to closing
df['age_closed_startup'] = df.apply(lambda row: (row.closed_at - row.founded_at) , axis=1)
# define age_closed_startup into age_startup_year
df['age_startup_year'] = df['age_closed_startup'].dt.days /365
# create a list of our conditions
conditions = [(df['relationships'] <= 5), (df['relationships'] > 5) & (df['relationships'] <= 10),
              (df['relationships'] > 10) & (df['relationships'] <= 16),(df['relationships'] > 16)]
# create a list of the values we want to assign for each condition
values = ['4', '3', '2', '1']
# create a new column tier_realtionships
df['tier_relationships'] = np.select(conditions, values)

# label encoder for startup status
le = LabelEncoder()
df['status_encoded'] = le.fit_transform(df['status'])
print(df)

# Drop Unused Column for Modelling
df = df.drop(['state_code'],axis=1)
df = df.drop(['id'],axis=1)
df = df.drop(['Unnamed: 6'],axis=1)
df = df.drop(['category_code'],axis=1)
df = df.drop(['object_id'],axis=1)
df = df.drop(['zip_code'],axis=1)
df = df.drop(['founded_at'],axis=1)
df = df.drop(['closed_at'],axis=1)
df = df.drop(['first_funding_at'],axis=1)
df = df.drop(['last_funding_at'],axis=1)
df = df.drop(['city'],axis=1)
df = df.drop(['name'],axis=1)
df = df.drop(['Unnamed: 0'],axis=1)
df = df.drop(['latitude','longitude'],axis=1)
df = df.drop(['age_closed_startup'],axis=1)
df = df.drop(['relationships'],axis=1)
df = df.drop(['status'],axis=1)
