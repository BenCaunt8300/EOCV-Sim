package com.github.serivesmejia.eocvsim.input;

import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import com.github.serivesmejia.eocvsim.EOCVSim;
import com.github.serivesmejia.eocvsim.gui.Visualizer;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import com.github.serivesmejia.eocvsim.util.Log;
import com.github.serivesmejia.eocvsim.util.SysUtil;

public class InputSourceManager {

	public volatile Mat lastMatFromSource = null;
	public volatile InputSource currInputSource = null;
	
	public volatile HashMap<String, InputSource> sources = new HashMap<>();

	public String currInputSourceName = "";

	private EOCVSim eocvSim;

	private volatile Map.Entry<String, InputSource> requestedToAddInputSource = null;
	private volatile boolean finishedWritingToAddInputSource = false;

	public volatile boolean finishedAddingRequestedSource = false;

	public InputSourceLoader inputSourceLoader = new InputSourceLoader();

	private volatile String nextInputSourceChange = "";

	public enum SourceType {
		IMAGE,
		CAMERA,
		UNKNOWN
	}

	public InputSourceManager(EOCVSim eocvSim) {
		this.eocvSim = eocvSim;
	}
	
	public void init() {

		Log.info("InputSourceManager", "Initializing...");

		inputSourceLoader.loadInputSourcesFromFile();

		for(Map.Entry<String, InputSource> entry : inputSourceLoader.loadedInputSources.entrySet()) {
			addInputSource(entry.getKey(), entry.getValue());
		}

		Size size = new Size(580, 480);
		createDefaultImgInputSource("/resources/images/ug_4.jpg", "ug_ocvsim_4.jpg", "Ultimate Goal 4 Ring", size);
		createDefaultImgInputSource("/resources/images/ug_1.jpg", "ug_ocvsim_1.jpg", "Ultimate Goal 1 Ring", size);
		createDefaultImgInputSource("/resources/images/ug_0.jpg", "ug_ocvsim_0.jpg", "Ultimate Goal 0 Ring", size);

		lastMatFromSource = new Mat();

		Log.white();
		
	}
	
	private void createDefaultImgInputSource(String resourcePath, String fileName, String sourceName, Size imgSize) {
		try {

			InputStream is = InputSource.class.getResourceAsStream(resourcePath);
			File f = SysUtil.copyFileIsTemp(is, fileName, true).file;

			ImageSource src = new ImageSource(f.getAbsolutePath(), imgSize);
			src.isDefault = true;

			addInputSource(sourceName, src);

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void setInputSourceNextFrame(String name) {
		this.nextInputSourceChange = name;
	}
	
	public void destroy() { }

	public void update() {

		if(requestedToAddInputSource != null && finishedWritingToAddInputSource) {
			addInputSource(requestedToAddInputSource.getKey(), requestedToAddInputSource.getValue());
			requestedToAddInputSource = null;
			finishedWritingToAddInputSource = false;
			finishedAddingRequestedSource = true;
		}

		if(!nextInputSourceChange.equals("")) {
			setInputSource(nextInputSourceChange);
			nextInputSourceChange = "";
		}

		if(currInputSource == null) return;
		lastMatFromSource = currInputSource.update();

	}

	public void addInputSource(String name, InputSource inputSource) {
		
		if(inputSource == null) {
			currInputSource = null;
			return;
		}

		if(sources.containsKey(name)) return;
		
		sources.put(name, inputSource);

		inputSourceLoader.saveInputSource(name, inputSource);
		inputSourceLoader.saveInputSourcesToFile();

		Log.info("InputSourceManager", "Adding InputSource " + inputSource.toString() + " (" + inputSource.getClass().getSimpleName() + ")");

	}

	public void deleteInputSource(String sourceName) {

		InputSource src = sources.get(sourceName);

		if(src == null) return;
		if(src.isDefault) return;

		sources.remove(sourceName);

		inputSourceLoader.deleteInputSource(sourceName);
		inputSourceLoader.saveInputSourcesToFile();

	}
	
	public void setInputSource(String sourceName) {

		if(currInputSource != null) {
			currInputSource.reset();
		}

		InputSource src = sources.get(sourceName);

		if(src != null) {
			src.reset();
		}

		//check if source type is a camera, and if so, create a please wait dialog
		Visualizer.AsyncPleaseWaitDialog apwdCam = checkCameraDialogPleaseWait(sourceName);

		if(src != null) src.init();

		//if there's a please wait dialog for a camera source, destroy it.
		if(apwdCam != null) {
			apwdCam.destroyDialog();
		}

		currInputSource = src;
		
		Log.info("InputSourceManager", "Set InputSource to " + currInputSource.toString() + " (" + src.getClass().getSimpleName() + ")");
		
	}

	public Visualizer.AsyncPleaseWaitDialog checkCameraDialogPleaseWait(String sourceName) {

		Visualizer.AsyncPleaseWaitDialog apwdCam = null;

		if(getSourceType(sourceName) == SourceType.CAMERA) {
			apwdCam = eocvSim.visualizer.asyncPleaseWaitDialog("Opening camera...", null, "Exit",
																new Dimension(300, 150), true);
			apwdCam.onCancel(new Runnable() {
				@Override
				public void run() {
					System.exit(0);
				}
			});
		}

		return apwdCam;

	}

	public SourceType getSourceType(String sourceName) {

		InputSource source = sources.get(sourceName);

		return getSourceType(source);

	}

	public static SourceType getSourceType(InputSource source) {

		switch(source.getClass().getSimpleName()) {
			case "ImageSource":
				return SourceType.IMAGE;
			case "CameraSource":
				return SourceType.CAMERA;
		}

		return SourceType.UNKNOWN;

	}


}
