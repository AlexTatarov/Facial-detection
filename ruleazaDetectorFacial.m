function [detectii, scoruriDetectii, imageIdx] = ruleazaDetectorFacial(parametri)
% 'detectii' = matrice Nx4, unde
%           N este numarul de detectii
%           detectii(i,:) = [x_min, y_min, x_max, y_max]
% 'scoruriDetectii' = matrice Nx1. scoruriDetectii(i) este scorul detectiei i
% 'imageIdx' = tablou de celule Nx1. imageIdx{i} este imaginea in care apare detectia i
%               (nu punem intregul path, ci doar numele imaginii: 'albert.jpg')

% Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din parametri.numeDirectorExempleTest
% Directorul cu numele parametri.numeDirectorExempleTest contine imagini ce
% pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
% date MIT+CMU dar si pentru alte imagini (imaginile realizate cu voi la curs+laborator).
% Functia 'suprimeazaNonMaximele' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
% Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.

% Functia voastra ar trebui sa calculeze pentru fiecare imagine
% descriptorul HOG asociat. Apoi glisati o fereastra de dimeniune paremtri.dimensiuneFereastra x  paremtri.dimensiuneFereastra (implicit 36x36)
% si folositi clasificatorul liniar (w,b) invatat poentru a obtine un scor. Daca acest scor este deasupra unui prag (threshold) pastrati detectia
% iar apoi mporcesati toate detectiile prin suprimarea non maximelor.
% pentru detectarea fetelor de diverse marimi folosit un detector multiscale
if parametri.antrenareCuExemplePuternicNegative == 1
    imgFiles = dir( fullfile( parametri.numeDirectorExempleTest, '*.jpg' ));
else
    imgFiles = dir( fullfile( parametri.numeDirectorExempleNegative, '*.jpg' ));
end
%initializare variabile de returnat
detectii = zeros(0,4);
scoruriDetectii = zeros(0,1);
imageIdx = cell(0,1);
imaginiFinale = cell(0,1);

for i = 1:length(imgFiles)
    fprintf('Rulam detectorul facial pe imaginea %s\n', imgFiles(i).name)
    
    if parametri.antrenareCuExemplePuternicNegative == 1
        img = imread(fullfile( parametri.numeDirectorExempleTest, imgFiles(i).name ));
    else
        img = imread(fullfile( parametri.numeDirectorExempleNegative, imgFiles(i).name ));
    end
    %img = imread(fullfile( parametri.numeDirectorExempleTest, imgFiles(i).name ));
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    %completati codul functiei in continuare
    
    nrRedimensionari = 3;
    scaraMarire = [1.1, 1.2, 1.3];
    scaraMicsorare = [0.9, 0.8, 0.7];
    
    detectiiMarire = zeros(0,4);
    scoruriMarire = zeros(0,1);
    indecsiMarire = cell(0,1);
    exemplePNMarire = cell(0,1);
    
    detectiiMicsorare = zeros(0,4);
    scoruriMicsorare = zeros(0,1);
    indecsiMicsorare = cell(0,1);
    exemplePNMicsorare = cell(0,1);
    
    for j = 1:nrRedimensionari
        %marirea imaginii
        coeficientMarire = scaraMarire(j);
        img_aux = imresize(img,coeficientMarire);
        
        descriptorHOGImagine = vl_hog(single(img_aux),parametri.dimensiuneCelulaHOG);
        pas = round(parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG);
        
        detectii_aux = zeros(0,4);
        scoruriDetectii_aux = zeros(0,1);
        indecsi_aux = cell(0,1);
        
        for k = 1:size(descriptorHOGImagine,1)-pas
            for m = 1:size(descriptorHOGImagine,2)-pas
                descriptorHOGCurent = descriptorHOGImagine(k:k-1+pas,m:m-1+pas,:);
                scor = descriptorHOGCurent(:)'*parametri.w+parametri.b;
                if scor > parametri.threshold
                    raport_x = (size(img,2)/size(img_aux,2));
                    raport_y = (size(img,1)/size(img_aux,1));
                    
                    if parametri.antrenareCuExemplePuternicNegative == 0
                        
                        imagineCurenta = uint8(img_aux((k-1)*parametri.dimensiuneCelulaHOG+1:(k-1)*parametri.dimensiuneCelulaHOG+parametri.dimensiuneFereastra,(m-1)*parametri.dimensiuneCelulaHOG+1:(m-1)*parametri.dimensiuneCelulaHOG+parametri.dimensiuneFereastra));
                        exemplePNMarire = [exemplePNMarire imagineCurenta];
                    end
                    
                    detectii_aux = [detectii_aux; ceil(((m-1)*parametri.dimensiuneCelulaHOG+1)*raport_x) ceil(((k-1)*parametri.dimensiuneCelulaHOG+1)*raport_y) ceil(((m-1)*parametri.dimensiuneCelulaHOG+parametri.dimensiuneFereastra)*raport_x) ceil(((k-1)*parametri.dimensiuneCelulaHOG+parametri.dimensiuneFereastra)*raport_y)];
                    scoruriDetectii_aux = [scoruriDetectii_aux scor];
                    indecsi_aux = [indecsi_aux imgFiles(i).name];
                end
            end
        end
        
        rezultate = [];
        if(size(detectii_aux,1) > 0)
            rezultate = eliminaNonMaximele(detectii_aux,scoruriDetectii_aux,size(img));
        end
        
        detectiiMarire = [detectiiMarire; detectii_aux(rezultate,:)];
        scoruriMarire = [scoruriMarire scoruriDetectii_aux(rezultate)];
        indecsiMarire = [indecsiMarire indecsi_aux(rezultate)];
        
        if parametri.antrenareCuExemplePuternicNegative == 0
            exemplePNMarire = [exemplePNMarire exemplePNMarire(rezultate)];
        end
        
        
        %micsorarea imaginii
        coeficientMicsorare = scaraMicsorare(j);
        img_aux = imresize(img,coeficientMicsorare);
        
        if (size(img_aux,1) >= 42 && size(img_aux,2) >=42)
            
            descriptorHOGImagine = vl_hog(single(img_aux),parametri.dimensiuneCelulaHOG);
            pas = round(parametri.dimensiuneFereastra/parametri.dimensiuneCelulaHOG);
            
            detectii_aux = zeros(0,4);
            scoruriDetectii_aux = zeros(0,1);
            indecsi_aux = cell(0,1);
            
            for k = 1:size(descriptorHOGImagine,1)-pas
                for m = 1:size(descriptorHOGImagine,2)-pas
                    
                    descriptorHOGCurent = descriptorHOGImagine(k:k-1+pas,m:m-1+pas,:);
                    scor = descriptorHOGCurent(:)'*parametri.w+parametri.b;
                    if scor > parametri.threshold
                        
                        raport_x = (size(img,2)/size(img_aux,2));
                        raport_y = (size(img,1)/size(img_aux,1));
                        
                        if parametri.antrenareCuExemplePuternicNegative == 0
                            %idxPN = idxPN + 1;
                            imagineCurenta = uint8(img((k-1)*parametri.dimensiuneCelulaHOG+1:(k-1)*parametri.dimensiuneCelulaHOG+parametri.dimensiuneFereastra,(m-1)*parametri.dimensiuneCelulaHOG+1:(m-1)*parametri.dimensiuneCelulaHOG+parametri.dimensiuneFereastra));
                            exemplePNMicsorare = [exemplePNMicsorare imagineCurenta];
                        end
                        
                        detectii_aux = [detectii_aux; ceil(((m-1)*parametri.dimensiuneCelulaHOG+1)*raport_x) ceil(((k-1)*parametri.dimensiuneCelulaHOG+1)*raport_y) ceil(((m-1)*parametri.dimensiuneCelulaHOG+parametri.dimensiuneFereastra)*raport_x) ceil(((k-1)*parametri.dimensiuneCelulaHOG+parametri.dimensiuneFereastra)*raport_y)];
                        scoruriDetectii_aux = [scoruriDetectii_aux scor];
                        indecsi_aux = [indecsi_aux imgFiles(i).name];
                    end
                end
            end
            
            rezultate = [];
            if(size(detectii_aux,1) > 0)
                rezultate = eliminaNonMaximele(detectii_aux,scoruriDetectii_aux,size(img));
            end
            
            detectiiMicsorare = [detectiiMicsorare; detectii_aux(rezultate,:)];
            scoruriMicsorare = [scoruriMicsorare scoruriDetectii_aux(rezultate)];
            indecsiMicsorare = [indecsiMicsorare indecsi_aux(rezultate)];
            
            if parametri.antrenareCuExemplePuternicNegative == 0
                exemplePNMicsorare = [exemplePNMicsorare exemplePNMicsorare(rezultate)];
            end
            
        end
          
    end
    
    %vedem ce am obtinut
    
    detectiiTemporare = zeros(0,4);
    scoruriDetectiiTemporare = zeros(0,1);
    imageIdxTemporare = cell(0,1);
    exemplePN = cell(0,1);
    
    detectiiTemporare = [detectiiMicsorare; detectiiMarire];
    scoruriDetectiiTemporare = [scoruriMicsorare scoruriMarire];
    imageIdxTemporare = [indecsiMicsorare indecsiMarire];
    
    if parametri.antrenareCuExemplePuternicNegative == 0
        exemplePN = [exemplePNMicsorare exemplePNMarire];
    end
    
    rezultate = [];
    if(size(detectiiTemporare,1) > 0)
        rezultate = eliminaNonMaximele(detectiiTemporare,scoruriDetectiiTemporare,size(img));
    end
    nrPuternicNegative = 0;
    detectii = [detectii; detectiiTemporare(rezultate) ];
    scoruriDetectii = [scoruriDetectii scoruriDetectiiTemporare(rezultate) ];
    imageIdx = [imageIdx imageIdxTemporare(rezultate) ];
    
    %for ii = 1:size(detectiiTemporare,1)
    %    if rezultate(i) == 1
    %        nrPuternicNegative = nrPuternicNegative + 1;
    %        detectii = [detectii; detectiiTemporare(ii) ];
    %        scoruriDetectii = [scoruriDetectii scoruriDetectiiTemporare(ii) ];
    %        imageIdx = [imageIdx imageIdxTemporare(ii) ];
    %    end
    %    
    %end
    
    %detectii = [detectii; detectiiTemporare(rezultate,:)];
    %scoruriDetectii = [scoruriDetectii scoruriDetectiiTemporare(rezultate)];
    %imageIdx = [imageIdx imageIdxTemporare(rezultate)];
    if parametri.antrenareCuExemplePuternicNegative == 0
        imaginiFinale = [imaginiFinale exemplePN(rezultate)];
        parametri.numeDirectorExemplePuternicNegative = fullfile(parametri.numeDirectorSetDate,'exemplePuternicNegative');
        disp('size(imaginiFinale)');
        disp(size(imaginiFinale,2));
        for ii = 1:size(imaginiFinale,2)
            fullFileName = fullfile('../data/exemplePuternicNegative/', ['imaginea' num2str(ii) '.jpg']);
            imagineCurenta = uint8(imaginiFinale{ii});
            if size(imagineCurenta,1) == 36 && size(imagineCurenta,2) == 36
                imwrite(imagineCurenta, fullFileName)
            end
        end
    end
    %exemplePuternicNegative = exemplePuternicNegative + sum(rezultate);
    %disp(nrPuternicNegative);
    
    
    
end




