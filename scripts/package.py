import zipfile
import os

def package_project():
    zip_filename = "qrtop_deploy.zip"
    include_files = [
        "server/app.py", 
        "scripts/init_db.py", 
        "server/static", 
        "server/templates", 
        "data/captchas"
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
        for item in include_files:
            if os.path.isfile(item):
                zip_ref.write(item)
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    for file in files:
                        zip_ref.write(os.path.join(root, file))
    
    print(f"Successfully created {zip_filename}")
    print("Upload this file to PythonAnywhere and run 'unzip qrtop_deploy.zip' in your console.")

if __name__ == "__main__":
    package_project()
