from wtforms import StringField, validators, Form, HiddenField


class AddPersonForm(Form):
    first_name = StringField('first_name', [validators.InputRequired()])
    second_name = StringField('second_name', [validators.InputRequired()])
    photo_hidden = HiddenField('photo_hidden', [validators.InputRequired()])

class EditPersonForm(Form):
    first_name = StringField('first_name', [validators.InputRequired()])
    second_name = StringField('second_name', [validators.InputRequired()])