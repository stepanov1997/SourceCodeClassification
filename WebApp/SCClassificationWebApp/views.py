from django.http import request
from django.shortcuts import render

# Create your views here.
from SCClassificationWebApp.services.linear_estimators import classificate
from SCClassificationWebApp.forms import ClassificateForm


def classification_endpoint(request):
    form = ClassificateForm(request.POST, request.FILES)

    if request.method == 'POST':
        text = form['file'].value()
        # try:
        classificated_class = classificate(text=text,
                                           type_of_estimator=form['estimator'].value(),
                                           type_of_vectorization=form['vectorization'].value(),
                                           representation_without_comments=form['with_comments'].value())
        # except NotImplementedError as e:
        if form.is_valid():
            return render(request, 'home.html', {
                'classificated_class': classificated_class,
                'form': form,
                'input_file': form['file'].name

            })
        #     'message': 'You should fill every form field.'
        # })
        else:
            return render(request, 'home.html', {
                'classificated_class': classificated_class,
                'form': form
            })

    else:
        return render(request, 'home.html', {'form': form})
