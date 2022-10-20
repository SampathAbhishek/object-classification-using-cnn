from django.shortcuts import render

from django.http import HttpResponseRedirect
from django.urls import reverse_lazy
from django.views.generic import TemplateView
from employee.forms import EmployeeForm

from django.views.generic import DetailView
from employee.models import Employee

class EmployeeImage(TemplateView):

form = EmployeeForm
template_name = &#39;emp_image.html&#39;

def post(self, request, *args, **kwargs):

form = EmployeeForm(request.POST, request.FILES)

if form.is_valid():
obj = form.save()

return HttpResponseRedirect(reverse_lazy(&#39;emp_image_display&#39;,
kwargs={&#39;pk&#39;: obj.id}))

context = self.get_context_data(form=form)
return self.render_to_response(context)

def get(self, request, *args, **kwargs):
return self.post(request, *args, **kwargs)

class EmpImageDisplay(DetailView):

model = Employee
template_name = &#39;emp_image_display.html&#39;
context_object_name = &#39;emp&#39;

def plant(request):

result1 = Employee.objects.latest(&#39;id&#39;)
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
models = keras.models.load_model(&#39;C:/Users/PIRO25/Desktop/ITPDL04
OBJECT CLASSIFICATION USING CNN/django
Deploy/employee/object_.h5&#39;)

from tensorflow.keras.preprocessing import image
test_image = image.load_img(&#39;C:/Users/PIRO25/Desktop/ITPDL04 OBJECT
CLASSIFICATION USING CNN/django Deploy/media/&#39;+ str(result1),
target_size=(250, 250))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = models.predict(test_image)
prediction = result[0]
prediction = list(prediction)
classes = [&#39;bike&#39;, &#39;cars&#39;, &#39;flowers&#39;, &#39;horse&#39;, &#39;human&#39;]
output = zip(classes, prediction)
output = dict(output)

if output[&#39;bike&#39;] == 1.0:

a = &#39;bike&#39;
elif output[&#39;cars&#39;] == 1.0:
a = &#39;cars&#39;
elif output[&#39;flowers&#39;] == 1.0:
a = &#39;flowers&#39;
elif output[&#39;horse&#39;] == 1.0:
a = &#39;horse&#39;
elif output[&#39;human&#39;] == 1.0:
a = &#39;human&#39;

return render(request, &quot;result.html&quot;, {&quot;out&quot;:a})
