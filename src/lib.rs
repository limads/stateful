use std::error::Error;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::cell::RefCell;
use std::boxed;

pub trait PersistentState<T> : Sized {

    /// Attempts to open UserState by deserializing it from a JSON path.
    /// This is a blocking operation.
    fn recover(path : &str) -> Option<Self>;

    fn update(&self, target : &T);

    // Saves the state to the given path by spawning a thread. This is
    // a nonblocking operation.
    fn persist(&self, path : &str) -> std::thread::JoinHandle<bool>;

}

/// Like Callbacks<A>, but defines both a set of arguments and a return type.
#[derive(Clone, Default)]
pub struct ValuedCallbacks<A, R>(Rc<RefCell<Vec<boxed::Box<dyn Fn(A)->R + 'static>>>>);

/// Holds a set of callbacks with a given signature. Useful if several objects must respond to
/// the same signal. If more than one object must be taken as argument, use a custom struct or
/// tuples.
pub type Callbacks<A> = ValuedCallbacks<A, ()>;

// pub type Callbacks<A> = Rc<RefCell<Vec<boxed::Box<dyn Fn(A) + 'static>>>>;

pub type BindResult<T> = Result<T, Box<dyn Error>>;

impl<A, R> ValuedCallbacks<A, R> {

    pub fn try_call(&self, args : A) -> BindResult<()>
    where
        A :  Clone
    {
        self.0.try_borrow()?.iter().for_each(|f| { f(args.clone()); } );
        Ok(())
    }

    pub fn call(&self, args : A)
    where
        A :  Clone
    {
        self.0.borrow().iter().for_each(|f| { f(args.clone()); });
    }

    pub fn try_call_with_values(&self, args : A) -> BindResult<Vec<R>>
    where
        A :  Clone,
        R : Sized
    {
        Ok(self.0.try_borrow()?.iter().map(|f| f(args.clone()) ).collect())
    }

    pub fn call_with_values(&self, args : A) -> Vec<R>
    where
        A :  Clone,
        R : Sized
    {
        self.0.borrow().iter().map(|f| f(args.clone()) ).collect()
    }

    pub fn try_bind(&self, f : impl Fn(A)->R + 'static) -> BindResult<()> {
        self.0.try_borrow_mut()?.push(boxed::Box::new(f));
        Ok(())
    }

    pub fn bind(&self, f : impl Fn(A)->R + 'static) {
        self.0.borrow_mut().push(boxed::Box::new(f));
    }

    pub fn try_count_bounded(&self) -> BindResult<usize> {
        Ok(self.0.try_borrow()?.len())
    }

    pub fn count_bounded(&self) -> usize {
        self.0.borrow().len()
    }

}

/// Generic trait to represent interactions between Views (widgets or sets of grouped widgets affected by data change),
/// Models (data structures that encapsulate data-modifying algorithms) and controls (widgets
/// that modify the data contained in models). Widgets that affect a model (Controls) are represented by having the model imlement React<Widget>.
/// Widgets that are affected by a model (Views) are represented by having the widget implement React<Model>.
/// The implementation will usually bind one or more closures to the argument. Usually, an widget is either a control OR a view
/// with respect to a given model, but might a assume both roles. A widget might also be a view for one model but the control for another model. Models
/// usually encapsulate a call to glib::Receiver::attach(.), waiting for messages that change their state. The models are implemented
/// by closures activated on "signals", implemented using Rust enums. The actual data is not held in the model structure, but is owned
/// by a single closure executing on the main thread whenever some signal enum is received. If required, the model might spawn new
/// threads or wait for response from worker threads, but they should never block.
pub trait React<S> {

    fn react(&self, source : &S);

}

/// Stateful objects for any reason can live at an invalid state. This trait simply
/// declares which conditions an object might not be valid, associating an Error
/// type to your algorithm.
pub trait Stateful {

    type Error : std::fmt::Display;

}

/// Attempts to initialize an object.
pub trait Initialize
where
    Self : Stateful + Sized
{

    fn initialize(&mut self) -> Result<Self, <Self as Stateful>::Error>;

}

/// Attempts to perform an operation after which the object will not be used anymore.
/// Taking the object by value guarantees it won't be used anymore.
pub trait Finalize
where
    Self : Stateful
{

    fn finalize(self) -> Result<(), <Self as Stateful>::Error>;

}

/// TODO move to verifiable crate, and make stateful dependent on it.
/// Trait implemented by stateful structs or enumerations for which
/// certain invariants must be held and can be verified at runtime.
/// You can #[derive(Verify)] if all fields of your structure satisfy
/// verify, in which case the error will be Box<dyn Error> wrapping the
/// first violation found.
pub trait Verify
where
    Self : Debug + Sized + Stateful
{

    fn verify(&self) -> Result<(), <Self as Stateful>::Error>;

    /// # Panic
    /// Panics when the structure is not in a valid state.
    fn assume_verified(&self) {
        match self.verify() {
            Err(e) => panic!("Invariant violated for {:?}: {}", self, e),
            _ => { },
        }
    }

    fn is_verified(&self) -> bool {
        self.verify().is_ok()
    }

    /// Applies the given closure, returning the object back to the user
    /// if it is at a valid state after the closure is applied to it, and
    /// consuming it if the closure left it at an invalid state.
    ///
    /// The example assume that at the default state the object is valid.
    /// # Example
    ///
    /// use std::mem;
    /// match value.apply(mem::take(&mut obj), |obj| *obj += 1 ) {
    ///    Ok(obj) => { *obj = obj },
    ///    Err(e) => { }
    /// }
    fn apply(mut self, mut f : impl FnMut(&mut Self)) -> Result<Self, <Self as Stateful>::Error> {
        f(&mut self);
        match self.verify() {
            Ok(_) => Ok(self),
            Err(e) => Err(e)
        }
    }

}

/*pub struct SliceError<T, E> {
    ix : usize,
    err : E
}

impl<E> Verify for [T]
where
    T : Verify<Error=E>
{
    fn verify(&self) -> Result<(), SliceError<E> {
        self.iter().map(|v| v.verify() )
    }
}*/

/// Trait implemented by structs whose state can be characterized as having one of a few
/// discrete states at any given time. stateful structs or enumerations (usually enumerations)
/// that can be at one of a few states known at compile time. Some of the transitions
/// from state T->U might be invalid and must be specified at the implementation
/// of self.transition. TODO rename to transition?
pub trait Transition
where
    Self : Debug + Stateful + Sized
{

    /// Calls Self::try_transition, panicking when the transition is not valid.
    fn transition(&mut self, to : Self) {
        // It would be nice to show the destination state, but we take to by value instead of by ref.
        match self.try_transition(to) {
            Err(e) => panic!("Invalid transition from {:?}: {}", self, e),
            _ => { }
        }
    }

    /// Modifies Self by setting it to another value at to. It might be the case
    /// that there isn't a valid transition between the current state and the state
    /// represented at to, in which case
    fn try_transition(&mut self, to : Self) -> Result<(), <Self as Stateful>::Error>;

    /// Checks if a transition is legal by verifying the current state and
    /// the next possible state before attempting the transition.
    /// If transition from state T->U is invalid, overwrite this method.
    fn verified_transition(
        &mut self,
        to : Self
    ) -> Result<(), <Self as Stateful>::Error>
    where
        Self : Verify
    {
        self.verify()?;
        to.verify()?;
        self.transition(to);
        Ok(())
    }

}

/*/// Assumes an operation is executed within the specified time. Perhaps
/// Best implemented via a macro to be applied over the function
/// call.
pub trait Timed {

    fn timed()

}*/


