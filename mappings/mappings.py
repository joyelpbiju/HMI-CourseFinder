# these mappings are  used to help in converting implicit queries into explicit queries

# Synonym mappings
synonym_mappings = {
    'summertime': 'summer',
    'hot season': 'summer',
    'warm season': 'summer',
    'sunny season': 'summer',
    'wintertime': 'winter',
    'cold season': 'winter',
    'frosty season': 'winter',
    'snowy season': 'winter',
    'Optional course': 'elective',
    'Choice subject': 'elective',
    'Select course': 'elective',
    'Non-compulsory class': 'elective',
    'Elective course': 'elective',
    'Elective subject': 'elective',
    'Free choice class': 'elective',
    'Alternative course': 'elective',
    'Voluntary course': 'elective',
    'Optional module': 'elective',
    'Selected study': 'elective',
    'Discretionary course': 'elective',
    'Chosen subject': 'elective',
    'Non-mandatory course': 'elective',
    'Mandatory': 'obligatory',
    'Compulsory': 'obligatory',
    'Required': 'obligatory',
    'Imperative': 'obligatory',
    'Essential': 'obligatory',
    'Necessary': 'obligatory',
    'Binding': 'obligatory',
    'Enforced': 'obligatory',
    'Inescapable': 'obligatory',
    'Required by law': 'obligatory',
    'Requisite': 'obligatory',
    'Prescribed': 'obligatory',
    'Non-optional': 'obligatory',
    'Unavoidable': 'obligatory',
    'Incumbent': 'obligatory'
}

# Keyword mappings for each facet
keyword_mappings = {
    'instructor': ['taught by', 'instructor', 'professor', 'lecturer', 'teacher', 'given by', 'conducted by', 'led by',
                   'facilitated by', 'presented by', 'offered by','headed by', 'course instructor', 'course teacher'],
    'title': ['about', 'find', 'related', 'subject', 'title', 'display','related to','course',' focused on', 'study','learn'],
    'term': ['term', 'semester', 'trimester', 'quarter', 'session', 'academic term', 'academic period', 'time of year',
             'academic session', 'school term', 'summertime', 'hot season', 'warm season', 'sunny season', 'wintertime',
             'cold season', 'frosty season', 'snowy season'],
    'lang': ['language', 'lang', 'medium of instruction', 'taught in', 'instruction language', 'course language',
             'language of instruction', 'language used', 'language spoken', 'german', 'english', 'Deutsch'],
    'duration': ['duration', 'length', 'course length', 'time period', 'time frame', 'length of the course',
                 'course duration', 'length of time', 'period of time', 'course period','duration of'],
    'content': ['content', 'course content', 'syllabus', 'topics covered', 'material', 'curriculum', 'course material',
                'course topics', 'subjects covered', 'areas covered'],
    'learning': ['learning objective', 'learning', 'learning goals', 'learning outcomes', 'objectives',
                 'course objectives', 'learning aims', 'educational objectives', 'course goals'],
    'credit': ['credit', 'credits', 'credit hours', 'ECTS', 'points', 'course credits', 'credit value',
               'number of credits', 'credit units'],
    'type': ['type', 'course type', 'type of course', 'kind of course', 'category', 'course category', 'type of class',
             'class type']
}

number_words = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}
