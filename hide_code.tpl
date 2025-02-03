{%- extends 'lab/index.html.j2' -%}

{% block input_group %}
    {%- if 'hide_code' not in cell.metadata.get('tags', []) -%}
        {{ super() }}
    {%- endif -%}
{% endblock input_group %}
