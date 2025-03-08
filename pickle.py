import pickle

# Save the classification model
with open('svm_model.pkl', 'wb') as files:
    pickle.dump(svm_model, files)
