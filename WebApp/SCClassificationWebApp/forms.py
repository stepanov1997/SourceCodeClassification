from django import forms


class LinearClassificateForm(forms.Form):
    file = forms.FileField(allow_empty_file=False)
    CHOICES_1 = [('knn', 'K-nearest neighbors algorithm (k-NN)'),
                 ('svm', 'Support vector machines (SVM)')]
    estimator = forms.ChoiceField(choices=CHOICES_1, widget=forms.RadioSelect)
    CHOICES_2 = [('binary', 'Binary vectorization'),
                 ('count', 'Count vectorization'),
                 ('tfidf', 'Tf-idf vectorization'), ]
    vectorization = forms.ChoiceField(choices=CHOICES_2, widget=forms.RadioSelect)
    with_comments = forms.BooleanField(required=False, initial=False, widget=forms.CheckboxInput)


class NeuralClassificateForm(forms.Form):
    file = forms.FileField(label="Upload source code file:",allow_empty_file=False)
    CHOICES_1 = [('dnn', 'Dense neural network (DNN)'),
                 ('cnn', 'Convolution neural network (CNN)')]
    estimator = forms.ChoiceField(label="Type of neural network:", choices=CHOICES_1, widget=forms.RadioSelect)
