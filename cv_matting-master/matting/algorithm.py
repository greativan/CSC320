## CSC320 Winter 2018
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv

# If you wish to import any additional modules
# or define other utility functions,
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################


#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing
    # algorithms. These images are initialized to None and populated/accessed by
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
        self._images = {
            'backA': None,
            'backB': None,
            'compA': None,
            'compB': None,
            'colOut': None,
            'alphaOut': None,
            'backIn': None,
            'colIn': None,
            'alphaIn': None,
            'compOut': None,
        }

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage
    def mattingInput(self):
        return {
            'backA':{'msg':'Image filename for Background A Color','default':None},
            'backB':{'msg':'Image filename for Background B Color','default':None},
            'compA':{'msg':'Image filename for Composite A Color','default':None},
            'compB':{'msg':'Image filename for Composite B Color','default':None},
        }
    # Same as above, but for the output arguments
    def mattingOutput(self):
        return {
            'colOut':{'msg':'Image filename for Object Color','default':['color.tif']},
            'alphaOut':{'msg':'Image filename for Object Alpha','default':['alpha.tif']}
        }
    def compositingInput(self):
        return {
            'colIn':{'msg':'Image filename for Object Color','default':None},
            'alphaIn':{'msg':'Image filename for Object Alpha','default':None},
            'backIn':{'msg':'Image filename for Background Color','default':None},
        }
    def compositingOutput(self):
        return {
            'compOut':{'msg':'Image filename for Composite Color','default':['comp.tif']},
        }

    # Copy the output of the triangulation matting algorithm (i.e., the
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.
    def useTriangulationResults(self):
        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    #########################################

    # Use OpenCV to read an image from a file and copy its contents to the
    # matting instance's private dictionary object. The key
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        img = cv.imread(fileName)

        self._images[key] = img
        success = True
        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 54-63.
    # See run() in run.py for usage examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        cv.imwrite(fileName, self._images[key])
        if self._images['compOut'] == None:
            success = False
            msg = "fail to write"
        success = True
        #########################################
        return success, msg

    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary
    # ojbect.
    def triangulationMatting(self):
        """
success, errorMessage = triangulationMatting(self)

        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        backA = self._images['backA'].astype('float64')
        backB = self._images['backB'].astype('float64')
        compA = self._images['compA'].astype('float64')
        compB = self._images['compB'].astype('float64')
        row = backA.shape[0]
        col = backA.shape[1]
        # comp = np.reshape(np.dstack((compA, compB)), (row, col, 6, 1))
        # back = np.reshape(np.dstack((backA, backB)), (row, col, 6, 1))
        # #identity manipulated
        # idt = np.tile(np.identity(3).flatten(), (row,col))
        # idt_new = idt.reshape((row, col, 3, 3))
        # idt_stack = np.dstack((idt_new, idt_new))
        # A = np.concatenate((idt_stack, (-1) * back), axis=3)
        # result = np.empty((row, col, 4, 1))
        # A_inv = np.linalg.pinv(A)
        # result = np.reshape(np.matmul(A_inv, (comp-back)), (row, col, 4))
        # self._images['colOut'] = np.clip(result[:,:,0:3], 0, 255)
        # self._images['alphaOut'] = np.clip(result[:,:,3], 0, 1) * 255

        r=(compA[:,:,2]-compB[:,:,2])*(backA[:,:,2]-backB[:,:,2])
        g=(compA[:,:,1]-compB[:,:,1])*(backA[:,:,1]-backB[:,:,1])
        b=(compA[:,:,0]-compB[:,:,0])*(backA[:,:,0]-backB[:,:,0])
        r_square = (backA[:,:,2]-backB[:,:,2])**2
        g_square = (backA[:,:,1]-backB[:,:,1])**2
        b_square = (backA[:,:,0]-backB[:,:,0])**2

        alpha = (1 - (r+g+b)/(r_square+g_square+b_square))
        alpha_shaped = np.clip(np.reshape(alpha, (row, col, 1)), 0, 1)
        self._images['alphaOut'] = alpha_shaped * 255
        self._images['colOut']= (compA/255 - backA/255 + alpha_shaped*(backA/255)) * 255
        success = True
        msg = "matting success"
        #########################################

        return success, msg


    def createComposite(self):
        """
success, errorMessage = createComposite(self)

        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
"""

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        colIn = self._images['colIn'].astype('float64')
        alphaIn = self._images['alphaIn'].astype('float64')
        backIn = self._images['backIn'].astype('float64')
        self._images['compOut'] = (colIn/255+backIn/255-(alphaIn)/255*(backIn/255)) * 255
        success = True
        #########################################

        return success, msg


