from django import forms


class ClassificateForm(forms.Form):
    file = forms.FileField(allow_empty_file=False)
    CHOICES_1 = [('knn', 'K-nearest neighbors algorithm (k-NN)'),
               ('svm', 'Support vector machines (SVM)')]
    estimator = forms.ChoiceField(choices=CHOICES_1, widget=forms.RadioSelect)
    CHOICES_2 = [('binary', 'Binary vectorization'),
                 ('count', 'Count vectorization'),
                 ('tfidf', 'Tf-idf vectorization'),]
    vectorization = forms.ChoiceField(choices=CHOICES_2, widget=forms.RadioSelect)
    with_comments = forms.BooleanField(required=False, initial=False, widget=forms.CheckboxInput)

