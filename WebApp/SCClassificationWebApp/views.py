from django.http import request
from django.shortcuts import render

# Create your views here.
from SCClassificationWebApp.services.linear_estimators import classificate as lin_classificate
from SCClassificationWebApp.services.neural_estimators import classificate as nn_classificate
from SCClassificationWebApp.forms import LinearClassificateForm, NeuralClassificateForm


def home(request):
    return render(request, 'home.html', {})

def linear_classification_endpoint(request):
    if request.method == 'POST':
        form = LinearClassificateForm(request.POST, request.FILES)
        text = form['file'].value()
        # try:
        classificated_class = lin_classificate(text=text,
                                           type_of_estimator=form['estimator'].value(),
                                           type_of_vectorization=form['vectorization'].value(),
                                           representation_without_comments=form['with_comments'].value())
        # except NotImplementedError as e:
        part = 'out' if form['with_comments'].value() else ""
        if form.is_valid():
            return render(request, 'classification_page.html', {
                'classificated_class': classificated_class,
                'form': form,
                'input_file': form.files['file'].name,
                'type': f"{form['estimator'].value().upper()}, {form['vectorization'].value()}, with{part} comments",
                'neural': False
            })
        #     'message': 'You should fill every form field.'
        # })
        else:
            return render(request, 'classification_page.html', {
                'classificated_class': classificated_class,
                'form': form,
                'neural': False
            })

    else:
        form = LinearClassificateForm()
        return render(request, 'classification_page.html', {'form': form, 'neural': False})


def neural_classification_endpoint(request):
    if request.method == 'POST':
        form = NeuralClassificateForm(request.POST, request.FILES)
        text = form['file'].value()
        # try:
        classificated_class = nn_classificate(text=text, type_of_neural_network=form['estimator'].value())
        # except NotImplementedError as e:
        if form.is_valid():
            return render(request, 'classification_page.html', {
                'classificated_class': classificated_class,
                'form': form,
                'input_file': form.files['file'].name,
                'type': f"{form['estimator'].value().upper()}",
                'neural': True

            })
        #     'message': 'You should fill every form field.'
        # })
        else:
            return render(request, 'classification_page.html', {
                'classificated_class': classificated_class,
                'form': form,
                'neural': True
            })

    else:
        form = NeuralClassificateForm()
        return render(request, 'classification_page.html', {'form': form, 'neural': True})

