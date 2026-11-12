from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

def train_baseline(df):
    X = df.drop(columns=['sequence_id', 'subject', 'gesture'])
    y = df['gesture']
    groups = df['subject']
    
    # Aynı kişinin verisi hem train hem val'e düşmesin diye GroupShuffleSplit kullanıyoruz
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    score = f1_score(y_val, preds, average='macro')
    return model, score, X_val, y_val, preds