from django.db import migrations, models


class Migration(migrations.Migration):
    """Additive only: new nullable/defaulted columns, no data changes."""

    dependencies = [
        ('ai_assistant', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='actionrequest',
            name='operation_id',
            field=models.CharField(blank=True, max_length=36, null=True, unique=True),
        ),
        migrations.AddField(
            model_name='actionrequest',
            name='thread_id',
            field=models.CharField(blank=True, default='', max_length=200),
        ),
        migrations.AddField(
            model_name='actionrequest',
            name='result',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='actionrequest',
            name='decided_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='actionrequest',
            name='status',
            field=models.CharField(choices=[('PENDING', 'Pending'), ('APPROVED', 'Approved'), ('REJECTED', 'Rejected'), ('EXECUTED', 'Executed'), ('FAILED', 'Failed')], default='PENDING', max_length=10),
        ),
    ]
