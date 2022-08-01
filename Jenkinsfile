
pipeline {

    
    agent any

    stages {

        stage("Download-data-build-test-lohrasb"){

             steps {

                                            sh '''
                                                 docker version
                                                 docker info
                                                 docker build -f Dockerfile.test -t build-image-test-lohrasb .
                                            '''


             }

        }

        stage("build-container-test-lohrasb") {
            

                 
             steps {

                                            

                                                sh '''
                                                 docker run build-image-test-lohrasb
                                                '''
                                            
            
                 }
            }

         
        
        stage("build-image-pypi-lohrasb") {
                 
             steps {

                                                 sh '''
                                                 docker version
                                                 docker info
                                                 docker build -f Dockerfile.publish -t build-image-pypi-lohrasb .
                                                 '''

            
                 }
            }
    
        stage("build-container-pypi-lohrasb") {
            

                 
             steps {

                 withCredentials([
                              usernamePassword(credentialsId: 'twine-login-info',
                              usernameVariable: 'username',
                              passwordVariable: 'password',
                              ),
                              usernamePassword(credentialsId: 'git-login-info-token	',
                              usernameVariable: 'gitusername',
                              passwordVariable: 'gitpassword',
                              )
                              
                                              ]) 

                                              {

                                                 sh '''
                                                 docker run --env username=${username} --env password=${password} --env gitusername=${gitusername}  --env gitpassword=${gitpassword} build-image-pypi-lohrasb
                                                 '''
                                              }
            
                 }
            }

    
}

}

