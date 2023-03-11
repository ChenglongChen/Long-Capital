import datetime


def get_diff_date(d, diff, date_format="%Y-%m-%d"):
    d = datetime.datetime.strptime(d, date_format) + datetime.timedelta(days=diff)
    return d.strftime(date_format)


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d")
