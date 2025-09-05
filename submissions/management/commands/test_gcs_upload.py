import traceback
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile

class Command(BaseCommand):
    help = 'Forces an import of the GCS storage backend to reveal hidden errors.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING("Attempting to DIRECTLY import and use the GCS backend..."))
        self.stdout.write("This test is designed to crash and show the real, hidden error.")

        try:
            # Step 1: Attempt the direct import. This is what Django fails to do silently.
            self.stdout.write("--> STEP 1: Importing GoogleCloudStorage from storages.backends.gcloud...")
            from storages.backends.gcloud import GoogleCloudStorage
            self.stdout.write(self.style.SUCCESS("    Import successful! The library seems to be installed correctly."))

            # Step 2: Attempt to create an instance of the class.
            self.stdout.write("--> STEP 2: Instantiating GoogleCloudStorage()...")
            storage_instance = GoogleCloudStorage()
            self.stdout.write(self.style.SUCCESS("    Instantiation successful! Settings are likely being read."))

            # Step 3: Attempt to use the instance to save a file.
            self.stdout.write("--> STEP 3: Using the instance to save a file...")
            file_content = b'This is a direct test.'
            file_name = 'direct_upload_test.txt'
            path = storage_instance.save(file_name, ContentFile(file_content))
            self.stdout.write(self.style.SUCCESS("    Save successful! The file should be in your bucket."))
            self.stdout.write(f"    URL: {storage_instance.url(path)}")

        except Exception as e:
            # This is the part we have been waiting for!
            self.stderr.write(self.style.ERROR("\n--- A HIDDEN ERROR WAS FOUND! ---"))
            self.stderr.write(f"The error is of type: {type(e).__name__}")
            self.stderr.write(f"The error message is: {e}")
            self.stderr.write("\n--- Full Traceback ---")
            # Print the entire traceback to the console, which will show the exact line of the failure.
            traceback.print_exc()
            self.stderr.write("----------------------")